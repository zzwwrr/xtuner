"""Fused-EP backend parity tests (UniEP / Liger CuTe EP) vs a pure-torch EP reference.

Multi-rank numerical-correctness tests. The fused backend (whole routed-expert
segment: dispatch -> experts -> combine, routing weights applied) is compared
against a straightforward differentiable reference built only from
``torch.distributed.all_to_all_single`` + per-expert fp32 loops, so *both*
forward outputs and *all* gradients (dX, dW1/W3, dW2, d(topk_weights)) are
checked for value, with the backend running bf16.

Launch (single node, 8 GPUs; UniEP backward requires ep_size in {4, 8})::

    MOE_BACKEND_TEST=uniep torchrun --standalone --nproc_per_node=8 \
        tests/module/moe_backend/test_fused_ep_parity.py
    MOE_BACKEND_TEST=liger torchrun --standalone --nproc_per_node=8 \
        tests/module/moe_backend/test_fused_ep_parity.py

Direct execution requires CUDA and the selected backend; missing dependencies fail the run.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_to_all_single as differentiable_all_to_all
from torch.distributed.device_mesh import init_device_mesh

BACKEND = os.environ.get("MOE_BACKEND_TEST", "uniep")  # uniep | liger
# At least two local experts on EP8, so gate/up slices exercise their strided
# layout rather than the accidentally contiguous single-expert special case.
H, I, E, K, T = 256, 256, 16, 2, 64


def _world_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and torch.cuda.is_available()


def _backend_available() -> bool:
    try:
        if BACKEND == "uniep":
            import triton_dist.function.nvidia.ep_moe_fused  # noqa: F401
        elif BACKEND == "liger":
            import liger_kernel.ops.cute.ops  # noqa: F401
        else:
            return False
    except Exception:
        return False
    return True


def _torch_ep_reference(x, topk_ids, topk_weights, w13, w2, group):
    """Differentiable fp32 EP MoE reference with plain collectives.

    w13: [E_local, 2I, H], w2: [E_local, H, I] on this rank (E_local = E/world).
    Returns combined [T, H] and owns grads for x / topk_weights / w13 / w2
    through ordinary autograd.
    """
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    e_local = E // world
    t, h = x.shape

    # which (token, k) slots land on this rank?
    owner = topk_ids // e_local  # [T, K]

    # send/recv counts per destination rank
    send_counts = torch.zeros(world, dtype=torch.long, device=x.device)
    for r in range(world):
        send_counts[r] = (owner == r).sum()
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)

    # Build send buffer deterministically: sort slots by destination rank
    flat_owner = owner.reshape(-1)
    sorted_idx = torch.argsort(flat_owner, stable=True)
    send_rows_idx = sorted_idx[: int(send_counts.sum())]
    # dedupe guard: rows where owner==rank stay local; every slot is sent once
    send_token = send_rows_idx // K
    send_k = send_rows_idx % K
    send_x = x[send_token]
    send_w = topk_weights[send_token, send_k]
    send_expert = topk_ids.reshape(-1)[send_rows_idx]

    recv_x = torch.empty(int(recv_counts.sum()), h, dtype=x.dtype, device=x.device)
    recv_w = torch.empty(int(recv_counts.sum()), dtype=x.dtype, device=x.device)
    recv_e = torch.empty(int(recv_counts.sum()), dtype=torch.long, device=x.device)
    recv_x = differentiable_all_to_all(recv_x, send_x.contiguous(), recv_counts.tolist(), send_counts.tolist(), group=group)
    recv_w = differentiable_all_to_all(recv_w, send_w.contiguous(), recv_counts.tolist(), send_counts.tolist(), group=group)
    dist.all_to_all_single(recv_e, send_expert.contiguous(), recv_counts.tolist(), send_counts.tolist(), group=group)
    recv_e = recv_e.long()
    recv_w32 = recv_w.float()

    # local expert compute (fp32 loop over local experts)
    local_out = recv_x.float() * 0 + (w13.sum() + w2.sum()) * 0
    local_expert = recv_e - rank * e_local
    for e in range(e_local):
        rows = (local_expert == e).nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() == 0:
            continue
        seg = recv_x[rows].float()
        gu = seg @ w13[e].float().t()
        g, u = gu.chunk(2, dim=-1)
        act = F.silu(g) * u
        local_out[rows] = act @ w2[e].float().t()
    local_out = (local_out * recv_w32.unsqueeze(-1)).to(x.dtype)

    # send back
    back_x = torch.empty_like(send_x)
    back_x = differentiable_all_to_all(back_x, local_out.contiguous(), send_counts.tolist(), recv_counts.tolist(), group=group)

    # scatter-add into combined output
    combined = x.new_zeros(t, h)
    combined.index_add_(0, send_token, back_x.to(combined.dtype))
    return combined


def main():  # pragma: no cover - multi-rank entry
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    torch.manual_seed(42 + rank)

    if not _backend_available():
        raise RuntimeError(f"Required backend {BACKEND} is not importable.")

    if world not in (4, 8):
        raise ValueError("This smoke test requires 4 or 8 GPUs on one host.")
    ep_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))

    # Build a real MoEBlock for parameter storage (native layout) and the session
    from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock

    native = MoEBlock(
        hidden_size=H,
        moe_intermediate_size=I,
        n_routed_experts=E,
        moe_bias=False,
        ep_mesh=ep_mesh,
        expert_tp_mesh=None,
        ep_tp_mesh=None,
        float8_cfg=None,
        moe_act_fn_cfg=MoEActFnConfig(),
    ).cuda().bfloat16()
    with torch.no_grad():  # deterministic weights across ranks
        for parameter_index, (name, parameter) in enumerate(native.named_parameters()):
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            generator = torch.Generator(device="cpu").manual_seed(7 + rank * 10 + parameter_index)
            local.copy_(torch.randn(local.shape, generator=generator).to(local.device) * 0.02)
    for p in native.parameters():
        p.requires_grad_(True)

    class _LayerShim:
        """Minimal layer facade for the backend sessions."""

        experts = native
        _moe_top_k = K

    from xtuner.v1.module.moe_backend import build_fused_ep_forward

    kwargs = dict(max_tokens=T, num_sm=64, num_hosts=1)
    session = build_fused_ep_forward(BACKEND, layer=_LayerShim, ep_group=group, **kwargs)

    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(T, E, device="cuda", dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    topk_w, topk_i = torch.topk(probs, K, dim=-1)
    # separate leaf tensors so the two graphs do not accumulate into one .grad
    topk_w_f = topk_w.detach().clone().requires_grad_(True)  # fused backend
    topk_w_r = topk_w.detach().clone().requires_grad_(True)  # reference

    # ---- fused backend (bf16 ABI) --------------------------------------
    x_f = x.detach().clone().requires_grad_(True)
    y_f = session(x_f, topk_i, topk_w_f)
    output_grad = torch.randn_like(y_f)
    y_f.backward(output_grad)

    # ---- fp32 reference -------------------------------------------------
    def local_tensor(tensor):
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    w13 = local_tensor(native.fused_w1w3.weight).detach().float().view(E // world, 2 * I, H).clone().requires_grad_(True)
    w2 = local_tensor(native.fused_w2.weight).detach().float().view(E // world, H, I).clone().requires_grad_(True)
    x_r = x.detach().float().clone().requires_grad_(True)
    y_r = _torch_ep_reference(x_r, topk_i, topk_w_r, w13, w2, group)
    y_r.backward(output_grad.float())

    # ---- compare ---------------------------------------------------------
    def report(name, a, b, tol=3e-2):
        a, b = a.detach().float(), b.detach().float()
        delta = a - b
        relative_l2 = (delta.norm() / b.norm().clamp_min(1e-12)).item()
        ok = torch.allclose(a, b, rtol=tol, atol=tol) and relative_l2 <= 0.02
        diff = delta.abs().max().item()
        print(f"[rank{rank}] {name}: {'OK' if ok else 'MISMATCH'} "
              f"max|d|={diff:.4e} relative_l2={relative_l2:.6f}")
        return ok

    all_ok = True
    all_ok &= report("forward Y", y_f, y_r.to(y_f.dtype))
    all_ok &= report("dX", x_f.grad, x_r.grad)
    # uniep returns d(topk_weights) in bf16 (documented upstream); loosen a bit
    all_ok &= report("d(topk_weights)", topk_w_f.grad, topk_w_r.grad, tol=5e-2)
    w13_grad_ref = w13.grad
    fused_grad = local_tensor(native.fused_w1w3.weight.grad)
    all_ok &= report("dW13", fused_grad.view(E // world, 2 * I, H), w13_grad_ref, tol=5e-2)
    all_ok &= report(
        "dW2",
        local_tensor(native.fused_w2.weight.grad).view(E // world, H, I),
        w2.grad,
        tol=5e-2,
    )

    print(f"[rank{rank}] RESULT: {'PASS' if all_ok else 'FAIL'} (backend={BACKEND})")
    dist.barrier(group)
    dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    if _world_ready() or os.environ.get("RANK") is not None:
        main()
    else:  # plain pytest without torchrun
        import pytest

        pytest.skip("fused-EP parity tests require torchrun (see module docstring)")
