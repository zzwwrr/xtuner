"""CPU-only checks for the Liger adapter w13 candidates (#1 split-cat, #2 no-grad strided).

Loads ``liger.py`` straight from its file so neither the XTuner package nor the
native Liger wheel is imported; no CUDA device is touched.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

LIGER_PY = Path(__file__).resolve().parents[3] / "xtuner/v1/module/moe_backend/liger.py"


def _load():
    spec = importlib.util.spec_from_file_location("liger_adapter_under_test", LIGER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


liger = _load()


def _baseline_split(w13, i):
    return w13[:, :i, :].contiguous(), w13[:, i:, :].contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 8, 16), (16, 2 * 768, 64)])
def test_split_cat_forward_bitwise(dtype, shape):
    e, two_i, h = shape
    torch.manual_seed(0)
    w13 = torch.randn(e, two_i, h, dtype=dtype)
    ref_g, ref_u = _baseline_split(w13, two_i // 2)
    g, u = liger._SplitW13.apply(w13, two_i // 2)
    for a, b in ((g, ref_g), (u, ref_u)):
        assert a.is_contiguous() and a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
                           b.view(torch.int16 if dtype == torch.bfloat16 else torch.int32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_split_cat_backward_equal_up_to_zero_sign(dtype):
    e, i, h = 8, 32, 24
    torch.manual_seed(1)
    base = torch.randn(e, 2 * i, h, dtype=dtype)
    d_gate = torch.randn(e, i, h, dtype=dtype)
    d_up = torch.randn(e, i, h, dtype=dtype)
    # include exact zeros of both signs to pin the documented difference
    d_gate[0, 0, :4] = torch.tensor([0.0, -0.0, 0.0, -0.0], dtype=dtype)
    d_up[1, 2, :4] = torch.tensor([-0.0, -0.0, 0.0, 0.0], dtype=dtype)

    w_ref = base.clone().requires_grad_(True)
    g, u = _baseline_split(w_ref, i)
    torch.autograd.backward([g, u], [d_gate, d_up])

    w_new = base.clone().requires_grad_(True)
    g2, u2 = liger._SplitW13.apply(w_new, i)
    torch.autograd.backward([g2, u2], [d_gate, d_up])

    expected = torch.cat([d_gate, d_up], dim=1)
    assert torch.equal(w_new.grad, expected)  # candidate is exactly the cat, including -0.0
    assert torch.equal(w_ref.grad, w_new.grad)  # numerically equal (-0.0 == 0.0)
    ivt = torch.int16 if dtype == torch.bfloat16 else torch.int32
    diff = w_ref.grad.view(ivt) != w_new.grad.view(ivt)
    assert diff.sum() == 4  # the four injected -0.0 entries
    # bit differences exist only where the candidate returns -0.0
    assert torch.all(w_new.grad[diff] == 0)
    assert torch.all(torch.signbit(w_new.grad[diff]))
    assert not torch.any(torch.signbit(w_ref.grad[diff]))


def test_split_cat_through_param_view_and_accumulation():
    # mirrors the adapter: parameter [E*2I, H] -> view [E, 2I, H] -> split; two steps accumulate
    e, i, h = 4, 16, 8
    torch.manual_seed(2)
    p_ref = torch.randn(e * 2 * i, h, dtype=torch.bfloat16, requires_grad=True)
    p_new = p_ref.detach().clone().requires_grad_(True)
    for step in range(2):
        dg, du = torch.randn(e, i, h, dtype=torch.bfloat16), torch.randn(e, i, h, dtype=torch.bfloat16)
        g, u = _baseline_split(p_ref.view(e, 2 * i, h), i)
        ((g.float() * dg.float()).sum() + (u.float() * du.float()).sum()).backward()
        g2, u2 = liger._SplitW13.apply(p_new.view(e, 2 * i, h), i)
        ((g2.float() * dg.float()).sum() + (u2.float() * du.float()).sum()).backward()
        assert torch.equal(p_ref.grad, p_new.grad), step


def test_split_cat_single_branch_grad():
    e, i, h = 2, 4, 4
    w = torch.randn(e, 2 * i, h, requires_grad=True)
    g, _u = liger._SplitW13.apply(w, i)
    g.sum().backward()
    assert torch.equal(w.grad[:, :i], torch.ones(e, i, h))
    assert torch.equal(w.grad[:, i:], torch.zeros(e, i, h))


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("XTUNER_LIGER_W13_SPLIT_CAT", raising=False)
    assert _load().W13_SPLIT_CAT is False
    monkeypatch.setenv("XTUNER_LIGER_W13_SPLIT_CAT", "1")
    assert _load().W13_SPLIT_CAT is True


# --------------------------------------------------------------------------- #2
import sys
import types
from types import SimpleNamespace


def _fake_moe_fused_factory(calls):
    """Stand-in for liger_kernel.ops.cute.ops.moe_fused with the same branch rule.

    Grad branch rejects non-contiguous gate/up exactly like
    ``LigerExpertParallelFusedMoEFunction.forward``; both branches compute a CPU
    reference MoE (SwiGLU experts, weighted top-k sum) from whatever views arrive.
    """

    def moe_fused(X, expert_indices, expert_weights, all_B, all_C, all_A, num_experts, top_k, pg=None):
        needs_grad = torch.is_grad_enabled() and any(
            t.requires_grad for t in (X, expert_weights, all_B, all_C, all_A)
        )
        calls.append(dict(needs_grad=needs_grad, b_contig=all_B.is_contiguous(), c_contig=all_C.is_contiguous(),
                          b_stride=all_B.stride(), c_ptr_off=all_C.data_ptr() - all_B.data_ptr()))
        if needs_grad and not (all_B.is_contiguous() and all_C.is_contiguous()):
            raise ValueError("Strided MoE gate/up weights are supported only when gradients are disabled")
        xf = X.float()
        y = torch.zeros_like(xf)
        for k in range(top_k):
            e = expert_indices[:, k].long()
            g = torch.einsum("th,tih->ti", xf, all_B[e].float())
            u = torch.einsum("th,tih->ti", xf, all_C[e].float())
            h = torch.nn.functional.silu(g) * u
            y = y + expert_weights[:, k : k + 1].float() * torch.einsum("ti,thi->th", h, all_A[e].float())
        return y.to(X.dtype)

    return moe_fused


@pytest.fixture
def fake_liger(monkeypatch):
    calls = []
    ops = types.ModuleType("liger_kernel.ops.cute.ops")
    ops.moe_fused = _fake_moe_fused_factory(calls)
    for name in ("liger_kernel", "liger_kernel.ops", "liger_kernel.ops.cute"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "liger_kernel.ops.cute.ops", ops)
    return calls


def _session(mod, e_local=4, i=8, h=16, k=2, requires_grad=True, seed=3):
    torch.manual_seed(seed)
    w13 = torch.nn.Parameter(torch.randn(e_local * 2 * i, h) * 0.1, requires_grad=requires_grad)
    w2 = torch.nn.Parameter(torch.randn(e_local * h, i) * 0.1, requires_grad=requires_grad)
    experts = SimpleNamespace(fused_w1w3=SimpleNamespace(weight=w13), fused_w2=SimpleNamespace(weight=w2))
    sess = SimpleNamespace(layer=SimpleNamespace(experts=experts), n_routed_experts=e_local,
                           ep_group=SimpleNamespace(size=lambda: 1), intermediate_size=i, hidden_size=h, top_k=k)
    return sess, w13, w2


def _inputs(e_local=4, h=16, k=2, t=10, seed=4):
    torch.manual_seed(seed)
    x = torch.randn(t, h)
    ids = torch.stack([torch.randperm(e_local)[:k] for _ in range(t)])
    w = torch.softmax(torch.randn(t, k), -1)
    return x, ids, w


@pytest.mark.parametrize("split_cat", [False, True])
def test_nograd_strided_passes_views_and_matches(monkeypatch, fake_liger, split_cat):
    mod = _load()
    sess, *_ = _session(mod)
    x, ids, w = _inputs()
    monkeypatch.setattr(mod, "W13_SPLIT_CAT", split_cat)

    monkeypatch.setattr(mod, "NOGRAD_STRIDED", False)
    with torch.no_grad():
        ref = mod.LigerCuteSession.__call__(sess, x, ids, w)
    monkeypatch.setattr(mod, "NOGRAD_STRIDED", True)
    with torch.no_grad():
        out = mod.LigerCuteSession.__call__(sess, x, ids, w)

    assert fake_liger[0]["b_contig"] and fake_liger[0]["c_contig"]
    last = fake_liger[1]
    assert not last["needs_grad"] and not last["b_contig"] and not last["c_contig"]
    assert last["b_stride"] == (2 * 8 * 16, 16, 1)  # expert stride of packed w13
    assert last["c_ptr_off"] == 8 * 16 * 4  # up half starts I rows later (fp32 bytes)
    assert torch.equal(out, ref)


def test_nograd_strided_keeps_copies_when_grad_needed(monkeypatch, fake_liger):
    mod = _load()
    sess, w13, _ = _session(mod)
    x, ids, w = _inputs()
    monkeypatch.setattr(mod, "NOGRAD_STRIDED", True)
    out = mod.LigerCuteSession.__call__(sess, x, ids, w)  # grad enabled, params require grad
    assert fake_liger[-1]["needs_grad"] and fake_liger[-1]["b_contig"] and fake_liger[-1]["c_contig"]
    out.sum().backward()
    assert w13.grad is not None


def test_nograd_strided_used_when_nothing_requires_grad(monkeypatch, fake_liger):
    # grad mode on but frozen params and plain inputs: moe_fused also takes the no-grad branch
    mod = _load()
    sess, *_ = _session(mod, requires_grad=False)
    x, ids, w = _inputs()
    monkeypatch.setattr(mod, "NOGRAD_STRIDED", True)
    mod.LigerCuteSession.__call__(sess, x, ids, w)
    assert not fake_liger[-1]["needs_grad"] and not fake_liger[-1]["b_contig"]


def test_nograd_flag_default_off(monkeypatch):
    monkeypatch.delenv("XTUNER_LIGER_NOGRAD_STRIDED", raising=False)
    assert _load().NOGRAD_STRIDED is False
    monkeypatch.setenv("XTUNER_LIGER_NOGRAD_STRIDED", "1")
    assert _load().NOGRAD_STRIDED is True
