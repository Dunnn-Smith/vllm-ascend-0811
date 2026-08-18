# DSV4 Confidence Head MRV1 Implementation

## Branch

```text
feature/conf_head_v2_for_dsv4_MRV1
```

## Overview

This branch contains the MRV1 implementation of the **DSpark Confidence Head** adaptation for DSV4.

The implementation is based on the design and implementation approach of the **dynamic target verify** mechanism introduced by the upstream vLLM PR:

```text
vLLM PR #47808
```

The upstream PR has already been merged into vLLM.

This branch follows the dynamic target verify design from that upstream implementation and adapts the corresponding confidence-head-related logic to the **vllm-ascend MRV1** codebase.

