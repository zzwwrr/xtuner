"""Liger CuTe EP backend — native CUTLASS+NVSHMEM fused MoE (forward+backward).

Replaces the whole routed-expert segment with Liger's ``liger_cute_kernels``
native EP MoE op via its reference autograd entry ``moe_fused``
(``LigerExpertParallelFusedMoEFunction``): one fused forward kernel
(dispatch + GEMM1 + SwiGLU + GEMM2 + combine over NVSHMEM symmetric memory)
and one fused backward kernel returning dX, d(all_B), d(all_C), d(all_A) and
dW (routing-weight gradient) in a single launch.

Mapping onto XTuner state
-------------------------
* tokens            ``hidden_states`` flattened to ``[T, H]`` (bf16)
* routing           ``topk_ids`` ``[T, K]`` int32; ``topk_weights`` cast to
                    **bf16** (the native ABI requires it — XTuner keeps fp32
                    internally, the cast node is differentiable so the bf16
                    dW flows back into the fp32 router path; one extra
                    rounding on the router gradient, documented in the plan)
* gate/up weights   ``fused_w1w3.weight`` ``[E_local*2I, H]`` viewed as
                    ``[E_local, 2I, H]`` then split; Liger wants gate and up
                    as *separate contiguous* ``[E_local, I, H]`` tensors for
                    backward, so each half gets one differentiable
                    ``.contiguous()`` copy per call (per layer per step —
                    the known layout cost from the integration plan).
* down weights      ``fused_w2.weight`` viewed as ``[E_local, H, I]`` —
                    already contiguous, no copy.
* shared experts    stay in XTuner (outside the replacement boundary).

Setup contract (upstream, one per process before the first forward)
-------------------------------------------------------------------
1. ``torch.cuda.set_device`` + an initialised NCCL process group (XTuner does
   this);
2. ``nvshmem.init_from_pg()`` — NVSHMEM bootstrap over the world group;
3. ``nvshmem.resolve_team(ep_group)`` — collective, cached per process group;
4. ``tvm_ffi.moe_configure_symmetric(max_tokens, H, E, K, num_pes,
   num_hosts, gpus_per_host)`` — static symmetric workspace sizing; must
   cover every later call (``T * K + E * 128`` slots), changing the topology
   afterwards raises.

Known upstream constraints (documented, not hidden)
---------------------------------------------------
* bf16 ABI: tokens, weights and routing weights are bf16; the routing-weight
  gradient comes back in bf16.
* shape gate: H % 256 == 0 and I % 256 == 0 (checked eagerly here with a
  clear message; GLM5.2 H=6144/I=2048 and Qwen3-30B-A3B H=2048/I=768 both
  pass, Mixtral-style tuned configs are irrelevant for the RDC path).
* multi-host MoE is *supported by the code* (num_hosts > 1) but has no
  published benchmark; single-node is the validated configuration.
* requires the separately built native wheel ``liger_cute_kernels``
  (CUTLASS + NVSHMEM + TVM FFI); importing this module without it raises a
  actionable ImportError only when the backend is actually selected.
"""

from __future__ import annotations

import os

import torch


class LigerCuteSession:
    """Per-process Liger CuTe EP session; cached per (layer, ep_group)."""

    _configured_keys: set[tuple] = set()

    def __init__(
        self,
        *,
        layer,
        ep_group,
        max_tokens: int | None = None,
        num_hosts: int = 1,
        num_sm: int = 64,  # accepted for kwargs uniformity; not used by Liger
    ):
        if ep_group is None or ep_group.size() == 1:
            raise ValueError(
                "Liger CuTe EP backend requires a multi-rank EP group (ep_size > 1); "
                "for ep_size == 1 keep moe_backend='native'."
            )
        experts = layer.experts
        if getattr(experts.fused_w1w3, "bias", None) is not None or getattr(
            experts.fused_w2, "bias", None
        ) is not None:
            raise ValueError("Liger CuTe EP backend does not support expert bias (moe_bias=True).")
        if experts.hidden_size % 256 != 0 or experts.intermediate_size % 256 != 0:
            raise ValueError(
                "Liger CuTe EP requires hidden_size % 256 == 0 and "
                f"moe_intermediate_size % 256 == 0; got H={experts.hidden_size}, "
                f"I={experts.intermediate_size}."
            )
        if experts.num_routed_experts % ep_group.size() != 0:
            raise ValueError(
                f"n_routed_experts ({experts.num_routed_experts}) must be divisible by "
                f"ep_size ({ep_group.size()})."
            )
        self.layer = layer
        self.ep_group = ep_group
        self.hidden_size = experts.hidden_size
        self.intermediate_size = experts.intermediate_size
        self.n_routed_experts = experts.num_routed_experts
        self.top_k = layer._moe_top_k
        self.max_tokens = max_tokens
        self.num_hosts = num_hosts

        self._ensure_runtime()

    # ------------------------------------------------------------------ setup
    def _ensure_runtime(self):
        key = (
            self.ep_group.size(),
            self.hidden_size,
            self.n_routed_experts,
            self.top_k,
            self.max_tokens,
            self.num_hosts,
        )
        if key in LigerCuteSession._configured_keys:
            return
        if LigerCuteSession._configured_keys:
            raise ValueError("Liger symmetric runtime is already configured differently; restart the process.")
        try:
            from liger_cute_kernels import nvshmem, tvm_ffi
        except Exception as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "Liger CuTe EP backend requires the separately built native wheel "
                "'liger_cute_kernels' (see repos/Liger-Kernel/liger_cute_kernels/README.md: "
                "CUTLASS + NVSHMEM + apache-tvm-ffi). Original import error: " f"{e!r}"
            ) from e

        max_tokens = self.max_tokens if self.max_tokens is not None else 8192
        ep_size = self.ep_group.size()
        num_hosts = self.num_hosts
        if num_hosts < 1:
            raise ValueError("num_hosts must be positive.")
        gpus_per_host = ep_size // num_hosts
        if num_hosts * gpus_per_host != ep_size:
            raise ValueError(
                f"num_hosts ({num_hosts}) must divide ep_size ({ep_size}) for the "
                "Liger symmetric-memory topology."
            )

        torch.cuda.set_device(torch.cuda.current_device())
        nvshmem.init_from_pg(self.ep_group)
        nvshmem.resolve_team(self.ep_group)  # collective; cached upstream
        tvm_ffi.moe_configure_symmetric(
            max_tokens=max_tokens,
            hidden_dim=self.hidden_size,
            max_num_experts=self.n_routed_experts,
            max_top_k=self.top_k,
            num_pes=ep_size,
            num_hosts=num_hosts,
            gpus_per_host=gpus_per_host,
        )
        LigerCuteSession._configured_keys.add(key)

    # ---------------------------------------------------------------- forward
    def __call__(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        """``hidden_states`` ``[T, H]`` bf16 -> weighted combined ``[T, H]`` bf16."""
        from liger_kernel.ops.cute.ops import moe_fused

        experts = self.layer.experts
        w13 = experts.fused_w1w3.weight
        w2 = experts.fused_w2.weight
        if hasattr(w13, "to_local"):
            w13 = w13.to_local()
        if hasattr(w2, "to_local"):
            w2 = w2.to_local()

        e_local = self.n_routed_experts // self.ep_group.size()
        w13 = w13.view(e_local, 2 * self.intermediate_size, self.hidden_size)
        w2 = w2.view(e_local, self.hidden_size, self.intermediate_size)

        # Differentiable copies: Liger's backward requires gate/up as separate
        # contiguous [E_local, I, H] tensors; autograd scatters the gradients
        # back into the fused XTuner parameter.
        all_b = w13[:, : self.intermediate_size, :].contiguous()  # gate
        all_c = w13[:, self.intermediate_size :, :].contiguous()  # up
        all_a = w2  # [E_local, H, I], already contiguous

        trace = os.environ.get("LIGER_TRACE") in ("1", "pre")
        if trace:
            print(f"LIGER rank={os.environ.get('RANK')} before_sync grad={torch.is_grad_enabled()}", flush=True)
            torch.cuda.synchronize()
            print(f"LIGER rank={os.environ.get('RANK')} enter_fused", flush=True)
        output = moe_fused(
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights.to(torch.bfloat16),  # native ABI is bf16 (documented)
            all_b,
            all_c,
            all_a,
            num_experts=self.n_routed_experts,
            top_k=self.top_k,
            pg=self.ep_group,
        )
        if os.environ.get("LIGER_TRACE") in ("1", "post"):
            torch.cuda.synchronize()
            print(f"LIGER rank={os.environ.get('RANK')} exit_fused", flush=True)
        return output
