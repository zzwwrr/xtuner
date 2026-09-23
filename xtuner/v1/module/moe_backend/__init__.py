"""Optional fused-EP MoE backends (UniEP / Liger CuTe EP).

These backends replace the *whole* routed-expert segment of
``MoEDecoderLayer._forward``::

    native:  dispatch_preprocess -> dispatch -> dispatch_postprocess
             -> MoEBlock (grouped GEMM x2 + SwiGLU)
             -> combine_preprocess -> combine -> combine_postprocess

    fused:   one fused call (router results in, weighted combined tokens out)

Selection is config-driven (``MoEConfig.moe_backend``), default ``"native"``
keeps the original execution path untouched.  Both backends are imported
lazily: a machine without their runtime dependencies still runs the native
path (and any other backend) unchanged.

Backward contract
-----------------
Each backend owns a complete ``torch.autograd.Function``:

=============  ==========================  =======================================
backend        fused segment               gradients produced
=============  ==========================  =======================================
``uniep``      dispatch+GEMM1+SwiGLU       dX, d(fc1 gate/up), d(fc2), d(topk
               (routing-weight scaled)     weights) — see ``uniep.py`` notes for
               +GEMM2+combine in one       the upstream bf16 down-cast of the
               Triton megakernel chain     routing-weight gradient
``liger``      dispatch+GEMM1+SwiGLU       single fused native kernel returns
               +GEMM2+combine (NVSHMEM     dX, d(all_B), d(all_C), d(all_A),
               symmetric memory)           dW (routing weights)
=============  ==========================  =======================================

The router itself (gating GEMM, softmax/top-k, weight normalisation) stays on
the XTuner side for every backend, so ``d(topk_weights)`` flows back into the
router exactly as in the native path.
"""

from __future__ import annotations

from typing import Literal

FusedEpBackend = Literal["native", "uniep", "liger"]

__all__ = ["FusedEpBackend", "build_fused_ep_forward"]


def build_fused_ep_forward(backend: str, *, layer, ep_group, **kwargs):
    """Create (or fetch the cached) per-process backend session for ``layer``.

    Returns a callable ``forward(hidden_states_2d, topk_ids, topk_weights)``
    producing the weighted combined output ``[T, H]`` (shared experts and the
    final residual/hidden_factor scaling stay in ``MoEDecoderLayer``).
    """
    if backend == "uniep":
        from .uniep import UniEpSession

        return UniEpSession(layer=layer, ep_group=ep_group, **kwargs)
    if backend == "liger":
        from .liger import LigerCuteSession

        return LigerCuteSession(layer=layer, ep_group=ep_group, **kwargs)
    raise ValueError(
        f"Unknown fused-EP MoE backend {backend!r}; expected one of 'uniep' / 'liger'."
    )
