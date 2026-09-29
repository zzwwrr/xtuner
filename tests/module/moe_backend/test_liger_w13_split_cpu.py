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
