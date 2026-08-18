# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost-aware adaptive verification for the Ascend DSpark path.

1. profile draft/target costs once during startup;
2. select a batch-wide draft budget from confidence probabilities, then
   distribute that budget across current requests on the NPU.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class StepTimingSample:
    """One startup dummy step, split into target and drafter elapsed time."""

    forward_ms: float
    drafter_ms: float
    num_target_tokens: int
    num_reqs: int
    full_graph: bool
    profile_curve: str


def _timing_event() -> torch.npu.Event:
    return torch.npu.Event(enable_timing=True)


@dataclass
class _StepTimingEvents:
    forward_start: torch.npu.Event = field(default_factory=_timing_event)
    forward_end: torch.npu.Event = field(default_factory=_timing_event)
    drafter_start: torch.npu.Event = field(default_factory=_timing_event)
    drafter_end: torch.npu.Event = field(default_factory=_timing_event)


class StepTimingCollector:
    """Collect NPU event timings only inside `collect`.

    The normal execute path pays only the Python `if` checks. Every dummy step
    owns independent events, so all samples are resolved behind one final event
    synchronization after the profiling block.
    """

    def __init__(self) -> None:
        self._collecting = False
        self._step: _StepTimingEvents | None = None
        self._batch = (False, 0, 0, "both")
        self._timed: list[
            tuple[_StepTimingEvents, tuple[bool, int, int, str]]
        ] = []

    @contextlib.contextmanager
    def collect(self) -> Iterator[list[StepTimingSample]]:
        samples: list[StepTimingSample] = []
        self._collecting = True
        try:
            yield samples
        finally:
            self._collecting = False
            timed, self._timed, self._step = self._timed, [], None

        if not timed:
            return

        # Waiting for the final drafter event also makes all preceding event
        # pairs readable.
        timed[-1][0].drafter_end.synchronize()
        samples.extend(
            StepTimingSample(
                forward_ms=events.forward_start.elapsed_time(events.forward_end),
                drafter_ms=events.drafter_start.elapsed_time(events.drafter_end),
                num_target_tokens=num_target_tokens,
                num_reqs=num_reqs,
                full_graph=full_graph,
                profile_curve=profile_curve,
            )
            for events, (
                full_graph,
                num_target_tokens,
                num_reqs,
                profile_curve,
            ) in timed
        )

    def record_batch(
        self,
        *,
        num_target_tokens: int,
        num_reqs: int,
        full_graph: bool,
        profile_curve: str,
    ) -> None:
        if self._collecting:
            self._batch = (
                full_graph,
                num_target_tokens,
                num_reqs,
                profile_curve,
            )

    def forward_start(self) -> None:
        if self._collecting:
            self._step = _StepTimingEvents()
            self._step.forward_start.record()

    def forward_end(self) -> None:
        if self._step is not None:
            self._step.forward_end.record()

    def drafter_start(self) -> None:
        if self._step is not None:
            self._step.drafter_start.record()

    def drafter_end(self) -> None:
        if self._step is not None:
            self._step.drafter_end.record()
            self._timed.append((self._step, self._batch))
            self._step = None


def _interpolate_cost_curve(
    curve: list[tuple[int, float]],
    limit: int,
    graph_capture_limit: int,
) -> np.ndarray:
    """Convert sparse profile points into an integer-indexed cost table.

    At or below `graph_capture_limit` the target graph pads execution to the
    next captured token count, so cost is a step function. Above the limit
    (and for the eager DSpark drafter) linear interpolation is used.
    """
    xs, ys = np.asarray(curve, dtype=np.float64).T
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ys = np.maximum.accumulate(ys)
    values = np.arange(limit + 1, dtype=np.float64)

    if graph_capture_limit <= 0:
        result = np.interp(values, xs, ys)
    else:
        idx = np.searchsorted(xs, values, side="left")
        result = ys[np.minimum(idx, len(xs) - 1)]
        # Crossing the graph limit is a real discontinuity. Interpolate only
        # between profile points that are both on the eager/piecewise tail.
        smooth = values > graph_capture_limit
        above = xs > graph_capture_limit
        if smooth.any() and above.any():
            result[smooth] = np.interp(values[smooth], xs[above], ys[above])

    if len(xs) > 1:
        before = values < xs[0]
        first_slope = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1)
        result[before] = np.maximum(
            0.0, ys[0] + first_slope * (values[before] - xs[0])
        )

        after = values > xs[-1]
        last_slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1)
        result[after] = ys[-1] + last_slope * (values[after] - xs[-1])

    return result


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_num_batched_tokens: int,
    graph_capture_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `draft_cost[num_reqs]` and `verify_cost[num_tokens]`.

    DSpark draft graphs are disabled on this Ascend path, so the draft curve is
    interpolated smoothly. Only target verification uses graph-padding steps.
    """
    draft_table = _interpolate_cost_curve(
        draft_curve, max_num_reqs, graph_capture_limit=0
    )
    verify_table = _interpolate_cost_curve(
        verify_curve,
        max_num_batched_tokens,
        graph_capture_limit=graph_capture_limit,
    )
    return (
        np.maximum(draft_table, 0.0),
        np.maximum(verify_table, 1e-6),
    )


def select_total_verify_budget(
    confidence_probs: np.ndarray,
    draft_cost_ms: np.ndarray,
    verify_cost_ms: np.ndarray,
    min_verify_tokens_per_req: int,
) -> int:
    """Choose the budget maximizing expected output tokens per millisecond.

    `confidence_probs[r, i]` is conditional confidence for draft position `i`.
    Its cumulative product is the probability that position `i` survives
    rejection sampling. The numerator for budget B is:

        num_requests + sum(the B admitted survival probabilities)

    The first term is the target/bonus token produced for every request.
    """
    num_reqs, num_steps = confidence_probs.shape
    if num_reqs == 0:
        return 0

    min_k = min(max(min_verify_tokens_per_req, 0), num_steps)
    survival = np.cumprod(
        np.clip(confidence_probs.astype(np.float64), 1e-6, 1.0),
        axis=1,
    )

    # Keep a safe prefix for every request, then globally rank the remainder.
    # Survival is non-increasing along each row, so global top-k still maps to
    # one continuous prefix per request.
    forced_budget = num_reqs * min_k
    forced_gain = float(survival[:, :min_k].sum())
    optional_scores = np.sort(survival[:, min_k:].reshape(-1))[::-1]
    expected_outputs = np.concatenate(
        (
            np.asarray([num_reqs + forced_gain], dtype=np.float64),
            num_reqs + forced_gain + np.cumsum(optional_scores),
        )
    )

    budgets = forced_budget + np.arange(expected_outputs.size)
    total_target_tokens = np.minimum(
        num_reqs + budgets,
        verify_cost_ms.shape[0] - 1,
    )
    draft_idx = min(num_reqs, draft_cost_ms.shape[0] - 1)
    costs = draft_cost_ms[draft_idx] + verify_cost_ms[total_target_tokens]
    best_extra = int(np.argmax(expected_outputs / np.maximum(costs, 1e-6)))
    return int(forced_budget + best_extra)


def allocate_verify_lengths(
    confidence_probs: torch.Tensor,
    total_budget: int,
    min_verify_tokens_per_req: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Allocate a batch budget to live requests on the NPU."""
    num_reqs, num_steps = confidence_probs.shape
    keep_lens = out[:num_reqs]
    if num_reqs == 0:
        return keep_lens

    min_k = min(max(min_verify_tokens_per_req, 0), num_steps)
    min_total = num_reqs * min_k
    max_total = num_reqs * num_steps
    total_budget = min(max(int(total_budget), min_total), max_total)
    keep_lens.fill_(min_k)

    extra_budget = total_budget - min_total
    candidate_cols = num_steps - min_k
    if extra_budget == 0 or candidate_cols == 0:
        return keep_lens

    survival = torch.cumprod(
        confidence_probs.float().clamp(min=1e-6, max=1.0),
        dim=1,
    )
    flat = survival[:, min_k:].reshape(-1)
    winners = flat.topk(extra_budget).indices
    winner_reqs = winners // candidate_cols
    keep_lens.scatter_add_(
        0,
        winner_reqs.to(torch.int64),
        torch.ones_like(winner_reqs, dtype=keep_lens.dtype),
    )
    return keep_lens


class AdaptiveVerificationManager:
    """Own startup cost curves and the per-step DSpark budget policy."""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        max_num_batched_tokens: int,
        num_speculative_tokens: int,
        device: torch.device,
        method_params: dict,
        pin_memory: bool,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_tokens = num_speculative_tokens
        self.device = device

        self.initial_verify_budget_per_req = int(
            method_params.get("initial_verify_budget_per_req", 5)
        )
        self.min_verify_tokens_per_req = int(
            method_params.get("min_verify_tokens_per_req", 1)
        )
        self.profile_replays = int(method_params.get("profile_replays", 5))
        self.profile_context_len = int(
            method_params.get("profile_context_len", 8192)
        )

        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        self._graph_capture_limit = 0
        self.last_total_budget = 0
        self._verify_lengths = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )

        # CPU budget selection consumes the preceding step's confidence copy.
        # Current live confidence remains on NPU for per-request allocation.
        shape = (max_num_reqs, num_speculative_tokens)
        self._stale_gpu = [
            torch.empty(shape, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        self._stale_cpu = [
            torch.empty(
                shape,
                dtype=torch.float32,
                device="cpu",
                pin_memory=pin_memory,
            )
            for _ in range(2)
        ]
        for buf in self._stale_cpu:
            buf.fill_(1.0)
        self._copy_events = [torch.npu.Event() for _ in range(2)]
        self._copy_stream = torch.npu.Stream(device=device)
        self._published_idx: int | None = None
        self._published_num_reqs = [0, 0]

    @property
    def is_profiled(self) -> bool:
        return self.cost_tables is not None

    def batches_to_profile(
        self,
        capture_sizes: list[int],
        decode_query_len: int,
        profile_context_len: int,
    ) -> Iterator[dict]:
        """Yield full-step dummy batches used to price draft and verification."""
        capture_sizes = sorted(
            {
                int(size)
                for size in capture_sizes
                if 0 < size <= self.max_num_batched_tokens
            }
        )
        self._graph_capture_limit = capture_sizes[-1] if capture_sizes else 0

        sizes = set(capture_sizes)
        if capture_sizes:
            # Add eager/piecewise points beyond the largest target graph.
            size = capture_sizes[-1]
            while size < self.max_num_batched_tokens:
                size = min(max(size + 1, size * 2), self.max_num_batched_tokens)
                sizes.add(size)
        else:
            # Eager-only deployment: profile a bounded geometric grid.
            size = min(max(decode_query_len, 1), self.max_num_batched_tokens)
            sizes.add(size)
            while size < self.max_num_batched_tokens:
                size = min(size * 2, self.max_num_batched_tokens)
                sizes.add(size)

        for num_tokens in sorted(sizes):
            for _ in range(self.profile_replays):
                yield {
                    "num_tokens": num_tokens,
                    "uniform_decode": True,
                    "force_attention": True,
                    "profile_seq_lens": profile_context_len,
                    "adaptive_profile_curve": "verify",
                }

        # Target capture sizes do not cover all possible request counts when
        # the maximum draft length is large. Price the eager DSpark drafter on
        # a separate geometric request grid. Here one target token is used per
        # request; only drafter_ms from these samples enters the cost table.
        num_reqs = 1
        while True:
            for _ in range(self.profile_replays):
                yield {
                    "num_tokens": num_reqs,
                    "uniform_decode": False,
                    "force_attention": True,
                    "profile_seq_lens": profile_context_len,
                    "adaptive_profile_curve": "draft",
                }
            if num_reqs == self.max_num_reqs:
                break
            num_reqs = min(num_reqs * 2, self.max_num_reqs)

    def set_initial_cost_curves(
        self,
        samples: list[StepTimingSample],
    ) -> None:
        def median_curve(
            points: Iterable[tuple[int, float]],
        ) -> list[tuple[int, float]]:
            grouped: defaultdict[int, list[float]] = defaultdict(list)
            for key, value in points:
                grouped[int(key)].append(float(value))
            return [
                (key, float(np.median(values)))
                for key, values in sorted(grouped.items())
            ]

        draft_curve = median_curve(
            (sample.num_reqs, sample.drafter_ms)
            for sample in samples
            if sample.profile_curve in ("draft", "both")
        )
        verify_curve = median_curve(
            (sample.num_target_tokens, sample.forward_ms)
            for sample in samples
            if sample.profile_curve in ("verify", "both")
        )
        self.set_cost_curves(draft_curve, verify_curve)

    def set_cost_curves(
        self,
        draft_curve: list[tuple[int, float]],
        verify_curve: list[tuple[int, float]],
    ) -> None:
        # Every TP rank must make the same CPU budget decision.
        draft_curve, verify_curve = get_tp_group().broadcast_object(
            (draft_curve, verify_curve), src=0
        )
        if not draft_curve or not verify_curve:
            raise RuntimeError(
                "DSpark adaptive verification could not profile step costs. "
                "Disable additional_config.dynamic_spec_config or check the "
                "startup dummy-run path."
            )

        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.max_num_reqs,
            self.max_num_batched_tokens,
            self._graph_capture_limit,
        )
        logger.info(
            "DSpark adaptive verification cost profile ready: "
            "%d draft points, %d verify points",
            len(draft_curve),
            len(verify_curve),
        )

    def _get_stale_confidences(
        self, num_reqs: int
    ) -> np.ndarray | None:
        idx = self._published_idx
        if idx is None:
            return None

        # Wait for the preceding step's copy, not current confidence compute.
        self._copy_events[idx].synchronize()
        stale_num_reqs = self._published_num_reqs[idx]
        stale = np.ones(
            (num_reqs, self.num_speculative_tokens), dtype=np.float32
        )
        copied = min(num_reqs, stale_num_reqs)
        if copied:
            stale[:copied] = self._stale_cpu[idx][:copied].numpy()
        return stale

    def _publish_confidences(
        self, confidence_probs: torch.Tensor
    ) -> None:
        num_reqs = confidence_probs.shape[0]
        write_idx = 0 if self._published_idx is None else self._published_idx ^ 1
        stage = self._stale_gpu[write_idx]
        stage[:num_reqs].copy_(confidence_probs)

        current_stream = torch.npu.current_stream(self.device)
        self._copy_stream.wait_stream(current_stream)
        with torch.npu.stream(self._copy_stream):
            self._stale_cpu[write_idx][:num_reqs].copy_(
                stage[:num_reqs], non_blocking=True
            )
            self._copy_events[write_idx].record()

        self._published_num_reqs[write_idx] = num_reqs
        self._published_idx = write_idx

    def update(
        self, confidence_probs: torch.Tensor
    ) -> torch.Tensor:
        """Choose a total budget from stale scores; allocate with live scores."""
        num_reqs = confidence_probs.shape[0]
        stale = self._get_stale_confidences(num_reqs)

        if stale is None or self.cost_tables is None:
            per_req = min(
                max(self.initial_verify_budget_per_req, 0),
                self.num_speculative_tokens,
            )
            total_budget = num_reqs * per_req
        else:
            draft_cost_ms, verify_cost_ms = self.cost_tables
            total_budget = select_total_verify_budget(
                stale,
                draft_cost_ms,
                verify_cost_ms,
                self.min_verify_tokens_per_req,
            )

        self._publish_confidences(confidence_probs)
        self.last_total_budget = total_budget
        return allocate_verify_lengths(
            confidence_probs,
            total_budget,
            self.min_verify_tokens_per_req,
            self._verify_lengths,
        )