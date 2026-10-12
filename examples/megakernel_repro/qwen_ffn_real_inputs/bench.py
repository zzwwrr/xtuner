"""Real-input routed FFN replay through XTuner native or Liger, on eight ranks."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
REPO = Path(__file__).resolve().parents[3]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('native', 'liger'), required=True)
    parser.add_argument('--tokens', type=int, choices=(8192, 16384), required=True)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(os.environ['REPRO_SPEC'])
    assert spec['backend'] == args.backend and spec['tokens'] == args.tokens
    assert spec['checkpoint'] and spec['rc'] == 1.0 and (not spec['compile'])
    assert Path(os.environ['BOARD_OUT']).resolve() == args.out.resolve()
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.utils.checkpoint import checkpoint
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=4))
    rank, world = (dist.get_rank(), dist.get_world_size())
    assert world == 8
    import xtuner, deep_ep, importlib.metadata
    assert Path(xtuner.__file__).resolve().is_relative_to(REPO)
    if rank == 0:
        Path(os.environ['BOARD_OUT'], 'runtime-paths.json').write_text(json.dumps(dict(xtuner=xtuner.__file__, deep_ep=deep_ep.__file__, torch=torch.__file__, versions={k: importlib.metadata.version(k) for k in ('torch', 'triton', 'deep-ep', 'liger-cute-kernels')}), indent=2))
    from xtuner.v1.ops import group_gemm
    assert group_gemm.__name__ == 'triton_group_gemm', group_gemm.__name__
    from xtuner.v1.utils import set_deterministic
    set_deterministic()
    torch.backends.cuda.matmul.allow_tf32 = False
    from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock
    from xtuner.v1.module.dispatcher import build_dispatcher
    from xtuner.v1.module.moe_backend import build_fused_ep_forward
    H, I, E, K = (2048, 768, 128, 8)
    T = spec['tokens']
    el = E // world
    out = Path(os.environ['BOARD_OUT'])
    mesh = init_device_mesh('cuda', (world,), mesh_dim_names=('ep',))
    reference = load('last_fp32_reference', REPO / 'tests/module/moe_backend/test_fused_ep_parity.py')
    reference.H, reference.I, reference.E, reference.K, reference.T = (H, I, E, K, T)
    segment = load('last_native_segment', REPO / 'tests/module/moe_backend/bench_moe_layer.py')._native_segment

    def local(t):
        return t.to_local() if hasattr(t, 'to_local') else t
    step, layer = (1, 23)
    f = args.capture_dir / f'replay-step{step}-layer{layer}-rank{rank}.pt'
    g = args.capture_dir / f'dy-step{step}-layer{layer}-rank{rank}.pt'
    data = torch.load(f, map_location='cpu', weights_only=True)
    x0, p0, ids = (data['x'].cuda(), data['prob'].cuda().float(), data['ids'].cuda().int())
    w13 = data['w13'].cuda().to(torch.bfloat16).view(el, 2 * I, H)
    w2 = data['w2'].cuda().to(torch.bfloat16).view(el, H, I)
    dy = torch.load(g, map_location='cpu', weights_only=True).cuda()
    del data
    provenance = dict(input=str(f), input_size=f.stat().st_size, dy=str(g), dy_size=g.stat().st_size)
    assert x0.shape == (T, H) and ids.shape == (T, K)
    assert x0.dtype == dy.dtype == torch.bfloat16
    (out / f'input-rank{rank}.json').write_text(json.dumps(provenance, indent=2))

    def block():
        b = MoEBlock(hidden_size=H, moe_intermediate_size=I, n_routed_experts=E, moe_bias=False, ep_mesh=mesh, expert_tp_mesh=None, ep_tp_mesh=None, float8_cfg=None, moe_act_fn_cfg=MoEActFnConfig()).cuda().bfloat16()
        with torch.no_grad():
            local(b.fused_w1w3.weight).copy_(w13.reshape(el * 2 * I, H))
            local(b.fused_w2.weight).copy_(w2.reshape(el * H, I))
        return b
    b = block()
    dispatcher = build_dispatcher(dispatcher='deepep', n_routed_experts=E, ep_group=dist.group.WORLD, tp_group=None, ep_tp_group=None, training_dtype='bf16', generate_dtype='bf16')

    def native_op(x, p, blk=b):
        return segment(x, dispatcher, blk, mesh, ids, p)
    if spec['backend'] == 'native':
        operation = native_op
    else:
        fused = build_fused_ep_forward(spec['backend'], layer=SimpleNamespace(experts=b, _moe_top_k=K), ep_group=dist.group.WORLD, max_tokens=T, num_sm=64, num_hosts=1)

        def operation(x, p):
            return fused(x, ids, p)

    def evaluate(op, blk):
        blk.zero_grad(set_to_none=True)
        x = x0.detach().clone().requires_grad_()
        p = p0.detach().clone().requires_grad_()
        y = checkpoint(op, x, p, use_reentrant=True) if spec['checkpoint'] else op(x, p)
        y.backward(dy)
        return dict(Y=y.detach().clone(), dX=x.grad.detach().clone(), dprob=p.grad.detach().clone(), dW13=local(blk.fused_w1w3.weight.grad).view(el, 2 * I, H).detach().clone(), dW2=local(blk.fused_w2.weight.grad).view(el, H, I).detach().clone())

    def metrics(actual, expected):
        result = {}
        for k, a in actual.items():
            a = a.float()
            z = expected[k].float()
            tol = 0.03 if k in ('Y', 'dX') else 0.05
            diff = a - z
            rel = float(diff.norm() / z.norm().clamp_min(1e-12))
            bad = int((diff.abs() > tol + tol * z.abs()).sum())
            finite = bool(torch.isfinite(a).all() and torch.isfinite(z).all())
            result[k] = dict(relative_l2=rel, max_abs=float(diff.abs().max()), bad_elements=bad, finite=finite, atol=tol, rtol=tol, passed=finite and bad == 0 and (rel <= 0.02))
        return result
    actual = evaluate(operation, b)
    xr = x0.float().requires_grad_()
    pr = p0.float().detach().requires_grad_()
    wr = w13.float().requires_grad_()
    vr = w2.float().requires_grad_()
    # FP32 reference communicates expert IDs as int64; backend IDs remain int32.
    yr = reference._torch_ep_reference(xr, ids.long(), pr, wr, vr, dist.group.WORLD)
    yr.backward(dy.float())
    errors = metrics(actual, dict(Y=yr, dX=xr.grad, dprob=pr.grad, dW13=wr.grad, dW2=vr.grad))
    del xr, pr, wr, vr, yr
    if spec['backend'] != 'native':
        nb = block()
        peer = evaluate(lambda x, p: native_op(x, p, nb), nb)
        peer_errors = metrics(actual, peer)
        del peer, nb
    else:
        peer_errors = {}
    passed = all((v['passed'] for v in list(errors.values()) + list(peer_errors.values())))
    (out / f'gate-rank{rank}.json').write_text(json.dumps(dict(fp32=errors, versus_native=peer_errors, passed=passed), indent=2))
    ok = torch.tensor(int(passed), device='cuda')
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    del actual
    b.zero_grad(set_to_none=True)
    if not ok.item():
        raise RuntimeError('Full output/gradient numerical gate failed; retain diagnostic evidence')
    b.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    records = []
    warmup = 10
    iters = 50
    for it in range(warmup + iters):
        x = x0.detach().clone().requires_grad_()
        p = p0.detach().clone().requires_grad_()
        b.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.synchronize()
        events = {}

        def forward(x, p):
            phase = ('Fr' if torch.is_grad_enabled() else 'F0') if spec['checkpoint'] else 'F'
            start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            start.record()
            with torch.cuda.nvtx.range(phase):
                y = operation(x, p)
            end.record()
            events[phase] = (start, end)
            return y
        import time
        tick = time.perf_counter()
        a, m, z = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        a.record()
        y = checkpoint(forward, x, p, use_reentrant=True) if spec['checkpoint'] else forward(x, p)
        m.record()
        with torch.cuda.nvtx.range('B_including_recompute' if spec['checkpoint'] else 'B'):
            y.backward(dy)
        z.record()
        torch.cuda.synchronize()
        if it >= warmup:
            row = dict(iteration=it - warmup, total_ms=a.elapsed_time(z), backward_including_recompute_ms=m.elapsed_time(z), wall_ms=(time.perf_counter() - tick) * 1000)
            row.update({k + '_ms': v[0].elapsed_time(v[1]) for k, v in events.items()})
            row['backward_excluding_recompute_estimate_ms'] = row['backward_including_recompute_ms'] - row.get('Fr_ms', 0)
            records.append(row)
    (out / f'rank{rank}.json').write_text(json.dumps(dict(rank=rank, spec=spec, records=records, max_allocated=torch.cuda.max_memory_allocated(), max_reserved=torch.cuda.max_memory_reserved(), scope='CUDA event routed FFN replay; backward includes checkpoint replay and internal rebuilding'), indent=2))
    dist.barrier()
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
