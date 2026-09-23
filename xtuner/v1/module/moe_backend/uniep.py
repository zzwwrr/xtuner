"""UniEP backend — Triton-distributed ``EpAll2AllFusedOp`` megakernel chain.

Replaces the whole routed-expert segment (dispatch -> experts -> combine) with
Triton-distributed's fused EP MoE (``mega_dispatch_group_gemm`` +
``mega_group_gemm_combine``), reusing its reference autograd Function
``TritonDistFusedEpMoeFunction`` so the backward pass (dX / dW1 / dW3 / dW2 /
dprob) is exactly the upstream one.

Mapping onto XTuner state
-------------------------
* tokens            ``hidden_states`` flattened to ``[T, H]`` (bf16)
* routing           ``topk_ids`` ``[T, K]`` int32, ``topk_weights`` ``[T, K]`` fp32
* fc1 gate/up       ``experts.fused_w1w3.weight`` ``[E_local*2I, H]`` viewed as
                    ``[E_local, 2I, H]`` and passed as packed fc1 with fc1_2=None.
                    This avoids concatenation in both forward and backward,
                    and lets autograd return one packed parameter gradient.
* fc2               ``experts.fused_w2.weight`` viewed as ``[E_local, H, I]``
                    (already contiguous, no copy).
* routing weights   applied inside the backend at SwiGLU time (row ``scale``),
                    mathematically equal to XTuner's unpermute-time weighting
                    (no expert bias), differing only in bf16 rounding order.

Known upstream constraints (documented, not hidden)
---------------------------------------------------
* single node only: the op asserts ``world_size == local_world_size`` and
  ``ep_group.size() <= 8``; backward additionally requires ep_size in {4, 8}.
* every MoE layer in the process must share one global op session (module
  level singleton in triton_dist), so all layers need identical H/I/E/K —
  true for a homogeneous MoE transformer.
* the upstream backward hard-casts the routing-weight gradient to bf16.
  The current upstream combine path synchronizes its stream in both forward
  and backward; dispatch also polls token counts on the CPU. Dynamic-routing
  CUDA Graph capture is unsupported by this host path.
* upstream 7908e4ea's SwiGLU backward requires the dA/dB row-offset fix.
  Packed weight passing does not repair that independent correctness bug;
  validate the installed dependency with the full output/gradient parity test.
* freezing ``fused_w2`` (requires_grad=False) trips an upstream
  ``UnboundLocalError``; we always train the experts, and guard here anyway.
"""

from __future__ import annotations

import torch

from . import build_fused_ep_forward  # noqa: F401  (re-export convenience)


class UniEpSession:
    """Per-process UniEP session; one instance is cached per (layer, ep_group)."""

    _initialized = False
    _ep_group = None
    _runtime_key = None

    def __init__(self, *, layer, ep_group, max_tokens: int | None = None, num_sm: int = 64, num_hosts: int = 1):
        if num_hosts != 1:
            raise ValueError("UniEP supports single-host EP only.")
        if ep_group is None or ep_group.size() == 1:
            raise ValueError(
                "UniEP backend requires a multi-rank EP group (ep_size > 1); "
                "for ep_size == 1 keep moe_backend='native'."
            )
        if ep_group.size() not in (4, 8):
            raise ValueError(
            f"UniEP training requires single-node EP4 or EP8, got ep_size={ep_group.size()}."
            )
        experts = layer.experts
        if getattr(experts.fused_w1w3, "bias", None) is not None or getattr(
            experts.fused_w2, "bias", None
        ) is not None:
            raise ValueError("UniEP backend does not support expert bias (moe_bias=True).")
        self.layer = layer
        self.ep_group = ep_group
        self.hidden_size = experts.hidden_size
        self.intermediate_size = experts.intermediate_size
        self.n_routed_experts = experts.num_routed_experts
        self.top_k = layer._moe_top_k
        self.max_tokens = max_tokens
        self.num_sm = num_sm

        self._ensure_runtime()

    # ------------------------------------------------------------------ setup
    def _ensure_runtime(self):
        """Idempotent one-time setup: NVSHMEM + the global ``EpAll2AllFusedOp``."""
        runtime_key = (self.hidden_size, self.intermediate_size, self.n_routed_experts,
                       self.top_k, self.max_tokens, self.num_sm)
        if UniEpSession._initialized:
            if UniEpSession._ep_group is not self.ep_group or UniEpSession._runtime_key != runtime_key:
                raise ValueError("UniEP runtime cannot be reused with a different group or configuration.")
            return
        try:
            from triton_dist.function.nvidia.common import init_triton_dist_ep_op
            from triton_dist.utils import init_nvshmem_by_torch_process_group
        except Exception as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "UniEP backend requires the 'triton_dist' package (Triton-distributed) "
                "with NVSHMEM4py installed and a CUDA device visible. "
                "Original import error: " f"{e!r}"
            ) from e

        max_tokens = self.max_tokens
        if max_tokens is None:
            # This is input tokens/rank. The op sizes expanded routing buffers
            # with top_k internally; multiplying here would apply it twice.
            max_tokens = 8192

        torch.cuda.set_device(torch.cuda.current_device())
        init_nvshmem_by_torch_process_group(self.ep_group)
        init_triton_dist_ep_op(
            self.ep_group,
            max_tokens,
            self.hidden_size,
            self.top_k,
            self.ep_group.rank(),
            self.n_routed_experts,
            self.ep_group.size(),
            dtype=torch.bfloat16,
            weight_dtype=torch.float32,
            num_sm=self.num_sm,
        )
        UniEpSession._initialized = True
        UniEpSession._ep_group = self.ep_group
        UniEpSession._runtime_key = runtime_key

    # ---------------------------------------------------------------- forward
    def __call__(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        """``hidden_states`` ``[T, H]`` bf16 -> weighted combined ``[T, H]`` bf16."""
        from triton_dist.function.nvidia.ep_moe_fused import TritonDistFusedEpMoeFunction

        experts = self.layer.experts
        w13 = experts.fused_w1w3.weight
        w2 = experts.fused_w2.weight
        # FSDP may hand us DTensor parameters; the fused op needs plain locals.
        if hasattr(w13, "to_local"):
            w13 = w13.to_local()
        if hasattr(w2, "to_local"):
            w2 = w2.to_local()

        e_local = self._experts_per_rank()
        # [E_local*2I, H] -> [E_local, 2I, H] (gate rows first, then up rows)
        w13 = w13.view(e_local, 2 * self.intermediate_size, self.hidden_size)
        # [E_local*H, I] -> [E_local, H, I]
        w2 = w2.view(e_local, self.hidden_size, self.intermediate_size)

        if not experts.fused_w2.weight.requires_grad and any(
            t.requires_grad for t in (hidden_states, topk_weights, w13)
        ):
            # Upstream bug: grad_fc2 unbound when fc2 frozen while others train.
            raise RuntimeError(
                "UniEP backward crashes when fused_w2 is frozen but other inputs "
                "require grad (upstream UnboundLocalError); unfreeze fused_w2 or "
                "freeze all expert weights."
            )

        return TritonDistFusedEpMoeFunction.apply(
            self.n_routed_experts,
            topk_weights,
            topk_ids.to(torch.int32),
            hidden_states,
            w13,
            None,
            w2,
            self.ep_group,
        )

    def _experts_per_rank(self) -> int:
        return self.n_routed_experts // self.ep_group.size()
