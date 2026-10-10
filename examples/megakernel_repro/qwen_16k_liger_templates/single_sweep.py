"""Sequential CPU-selected, GPU-measured 16K Liger template sweep.

Use torchrun with eight ranks. No process is launched by importing this file.
First sweep compiled FWD rows with current BWD, then compiled BWD rows with the
best FWD, and check the small Cartesian set of the top two FWD/top three BWD.
Every call completes checkpoint F0/recompute/backward before templates change.
Numerical gates compare the identical frozen real input with current CS2; this
is a screening reference, not an FP32 correctness or training-convergence gate.
"""
import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

BASE_FWD = '128,64,4,64,128,64,4,64,4,2,128'
BASE_BWD = '16,128,64,4,256,128,64,2,32,64,64,3,128'
SOURCE = Path(__file__).resolve().parents[3]
COMMIT = 'cb005e65fa54f9ee56a55778aeffa4bba9eed2af'
ROOT = Path(__file__).resolve().parent
DEFAULT_INVENTORY = ROOT / 'template_inventory.json'


def menu(inventory):
    data = json.loads(Path(inventory).read_text())['template_inventory']
    fwd, bwd = data['forward'], data['backward']
    assert BASE_FWD in fwd and BASE_BWD in bwd
    # Host dispatch maps legacy TM64 to GEMM M64 while communication/metadata
    # remain M128. Retain the exact installed tuple; all ten FWD rows match the
    # communication M128 of the fifteen BWD rows. The observed native chosen
    # TileM is recorded below as a runtime check of this ABI interpretation.
    compatible_fwd = list(fwd)
    compatible_bwd = [v for v in bwd if int(v.split(',')[-1]) == 128]
    skipped = []
    return compatible_fwd, compatible_bwd, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument('--case-step', type=int, default=1)
    parser.add_argument('--case-layer', type=int, default=23)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    fwd_menu, bwd_menu, skipped = menu(args.inventory)
    assert args.warmup >= 2 and args.iterations >= 5
    if args.cpu_check:
        print(json.dumps(dict(status='CPU_MENU_PASS', forward=len(fwd_menu),
                              backward=len(bwd_menu), skipped=skipped)))
        return
    assert args.capture_dir and args.out
    out = args.out.resolve()
    approved = Path(os.environ['REPRO_SUITE_OUTPUT']).resolve()
    assert out.is_relative_to(approved) and args.capture_dir.resolve().is_relative_to(approved)
    out.mkdir(parents=True, exist_ok=True)
    for rank in range(8):
        receipt = json.loads((args.capture_dir / f'capture-rank{rank}.json').read_text())
        assert receipt['status'] == 'COMPLETE' and receipt['same_call'] is True
        assert receipt['step'] == args.case_step and receipt['layer'] == args.case_layer
        assert receipt['shape'] == receipt['gradient_shape'] == [16384, 2048]
    subprocess.check_call(['git', '-C', str(SOURCE), 'merge-base', '--is-ancestor', COMMIT, 'HEAD'])
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    assert os.environ.get('LIGER_MOE_SM90_NONRDC') == '1'
    assert os.environ.get('XTUNER_LIGER_W13_SPLIT_CAT') == '1'
    assert os.environ.get('XTUNER_LIGER_NOGRAD_STRIDED', '0') == '0'
    assert os.environ.get('LIGER_DIAG_NO_K_SPLIT', '0') == '0'

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.utils.checkpoint import checkpoint
    import xtuner
    assert Path(xtuner.__file__).resolve().is_relative_to(SOURCE)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=4))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    from xtuner.v1.utils import set_deterministic
    from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock
    from xtuner.v1.module.moe_backend import build_fused_ep_forward
    from liger_cute_kernels import tvm_ffi
    set_deterministic()
    torch.backends.cuda.matmul.allow_tf32 = False
    H, I, E, K, T, local_experts = 2048, 768, 128, 8, 16384, 16
    stem = f'step{args.case_step}-layer{args.case_layer}-rank{rank}.pt'
    replay = args.capture_dir / ('replay-' + stem)
    dy_path = args.capture_dir / ('dy-' + stem)
    data = torch.load(replay, map_location='cpu', weights_only=True)
    x0 = data['x'].cuda().contiguous()
    p0 = data['prob'].cuda().float().contiguous()
    ids = data['ids'].cuda().int().contiguous()
    w13 = data['w13'].cuda().bfloat16().view(local_experts, 2 * I, H)
    w2 = data['w2'].cuda().bfloat16().view(local_experts, H, I)
    dy = torch.load(dy_path, map_location='cpu', weights_only=True).cuda().contiguous()
    del data
    assert x0.shape == dy.shape == (T, H) and p0.shape == ids.shape == (T, K)
    assert x0.dtype == dy.dtype == torch.bfloat16
    assert all(bool(torch.isfinite(v).all()) for v in (x0, p0, w13, w2, dy))
    mesh = init_device_mesh('cuda', (world,), mesh_dim_names=('ep',))
    block = MoEBlock(hidden_size=H, moe_intermediate_size=I, n_routed_experts=E,
                     moe_bias=False, ep_mesh=mesh, expert_tp_mesh=None, ep_tp_mesh=None,
                     float8_cfg=None, moe_act_fn_cfg=MoEActFnConfig()).cuda().bfloat16()
    def local(t):
        return t.to_local() if hasattr(t, 'to_local') else t
    with torch.no_grad():
        local(block.fused_w1w3.weight).copy_(w13.reshape(local_experts * 2 * I, H))
        local(block.fused_w2.weight).copy_(w2.reshape(local_experts * H, I))
    del w13, w2
    fused = build_fused_ep_forward('liger', layer=SimpleNamespace(experts=block, _moe_top_k=K),
                                  ep_group=dist.group.WORLD, max_tokens=T, num_hosts=1, num_sm=64)

    # Observe the native stack API without changing allocation or ownership.
    # Each no-grad F0 and grad Fr/B must return to depth zero. A failed native
    # call aborts this run; it is unsafe to continue a distributed stack blindly.
    stack = {'depth': 0, 'max_depth': 0, 'pushes': 0, 'pops': 0, 'tile_m': set()}
    native_fwd, native_pop = tvm_ffi.moe_fused_fwd_bf16, tvm_ffi.moe_pop_fwd
    def tracked_fwd(*a, **kw):
        value = native_fwd(*a, **kw)
        stack['depth'] += 1
        stack['pushes'] += 1
        stack['max_depth'] = max(stack['max_depth'], stack['depth'])
        stack['tile_m'].add(int(value[-1]))
        assert int(value[-1]) == 128, 'All installed templates must keep communication metadata at M128'
        return value
    def tracked_pop():
        assert stack['depth'] > 0
        value = native_pop()
        stack['depth'] -= 1
        stack['pops'] += 1
        return value
    tvm_ffi.moe_fused_fwd_bf16, tvm_ffi.moe_pop_fwd = tracked_fwd, tracked_pop
    def operation(x, p):
        return fused(x, ids, p)

    def evaluate():
        assert stack['depth'] == 0
        block.zero_grad(set_to_none=True)
        x, p = x0.detach().clone().requires_grad_(), p0.detach().clone().requires_grad_()
        y = checkpoint(operation, x, p, use_reentrant=True)
        y.backward(dy)
        torch.cuda.synchronize()
        assert stack['depth'] == 0
        return dict(Y=y.detach().clone(), dX=x.grad.detach().clone(), dprob=p.grad.detach().clone(),
                    dW13=local(block.fused_w1w3.weight.grad).detach().clone(),
                    dW2=local(block.fused_w2.weight.grad).detach().clone())

    def errors(actual, reference):
        result = {}
        for name, value in actual.items():
            a, b = value.float(), reference[name].float()
            delta = a - b
            tolerance = .03 if name in ('Y', 'dX') else .05
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            relative_l2 = float(delta.norm() / b.norm().clamp_min(1e-12))
            bad = int((delta.abs() > tolerance + tolerance * b.abs()).sum())
            result[name] = dict(finite=finite, relative_l2=relative_l2,
                                max_abs=float(delta.abs().max()), bad_elements=bad,
                                atol=tolerance, rtol=tolerance,
                                passed=finite and bad == 0 and relative_l2 <= .02)
        return result

    def select(fwd, bwd):
        assert stack['depth'] == 0
        torch.cuda.synchronize()
        dist.barrier()
        os.environ['LIGER_MOE_FORCE_CONFIG'] = fwd
        os.environ['LIGER_MOE_BWD_FORCE_CONFIG'] = bwd

    select(BASE_FWD, BASE_BWD)
    reference = evaluate()
    entries = []
    seen = {}

    def run(fwd, bwd, phase):
        key = fwd + '|' + bwd
        if key in seen:
            return seen[key]
        select(fwd, bwd)
        actual = evaluate()
        numerical = errors(actual, reference)
        del actual
        local_pass = all(v['passed'] for v in numerical.values())
        flag = torch.tensor(int(local_pass), device='cuda')
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        passed = bool(flag.item())
        rows = []
        if passed:
            for iteration in range(args.warmup + args.iterations):
                block.zero_grad(set_to_none=True)
                x, p = x0.detach().clone().requires_grad_(), p0.detach().clone().requires_grad_()
                dist.barrier()
                torch.cuda.synchronize()
                events = {}
                def timed(x, p):
                    name = 'Fr' if torch.is_grad_enabled() else 'F0'
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    y = operation(x, p)
                    end.record()
                    events[name] = (start, end)
                    return y
                start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                start.record()
                y = checkpoint(timed, x, p, use_reentrant=True)
                middle.record()
                y.backward(dy)
                end.record()
                torch.cuda.synchronize()
                assert stack['depth'] == 0
                if iteration >= args.warmup:
                    row = dict(iteration=iteration-args.warmup, total_ms=start.elapsed_time(end),
                               backward_including_recompute_ms=middle.elapsed_time(end))
                    row.update({name + '_ms': a.elapsed_time(b) for name, (a, b) in events.items()})
                    row['backward_excluding_recompute_ms'] = row['backward_including_recompute_ms'] - row['Fr_ms']
                    rows.append(row)
                del x, p, y, events
        block.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.synchronize()
        assert stack['depth'] == 0 and stack['pushes'] == stack['pops']
        item = dict(index=len(entries), phase=phase, fwd=fwd, bwd=bwd, gate_pass=passed,
                    numerical=numerical, records=rows, stack=dict(stack, tile_m=sorted(stack['tile_m'])),
                    memory=dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved()))
        with (out / f'sweep-rank{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(item) + '\n')
        gathered = [None] * world
        dist.all_gather_object(gathered, item)
        total_ms = (statistics.mean(max(v['records'][i]['total_ms'] for v in gathered)
                                    for i in range(args.iterations)) if passed else None)
        aggregate = dict(index=item['index'], phase=phase, fwd=fwd, bwd=bwd,
                         gate_pass=passed, total_ms=total_ms,
                         worst_relative_l2=max(z['relative_l2'] for v in gathered for z in v['numerical'].values()))
        entries.append(aggregate)
        seen[key] = aggregate
        if rank == 0:
            print(json.dumps(aggregate), flush=True)
            (out / 'sweep-progress.json').write_text(json.dumps(dict(entries=entries), indent=2))
        return aggregate

    baseline = run(BASE_FWD, BASE_BWD, 'baseline_repeat')
    assert baseline['gate_pass'], 'Current template self-repeat did not satisfy screening gate'
    first = [run(fwd, BASE_BWD, 'forward_menu') for fwd in fwd_menu]
    first = sorted((v for v in first if v['gate_pass']), key=lambda v: v['total_ms'])
    assert first
    second = [run(first[0]['fwd'], bwd, 'backward_menu') for bwd in bwd_menu]
    second = sorted((v for v in second if v['gate_pass']), key=lambda v: v['total_ms'])
    assert second
    for fwd in first[:2]:
        for bwd in second[:3]:
            run(fwd['fwd'], bwd['bwd'], 'shortlist_cross')
    candidates = sorted((dict(v, ratio_vs_baseline=baseline['total_ms']/v['total_ms'])
                         for v in entries if v['gate_pass'] and
                         (v['fwd'], v['bwd']) != (BASE_FWD, BASE_BWD)), key=lambda v: v['total_ms'])
    summary = dict(status='COMPLETE', scope='EP8 16K frozen real input; checkpoint F0 + recompute + backward; layer screening only',
                   source=str(SOURCE), commit=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(), capture_dir=str(args.capture_dir.resolve()),
                   case=[args.case_step, args.case_layer], reference='Current CS2/BWD template on identical input; not FP32 or convergence acceptance',
                   timing='Mean over iterations of slowest-rank CUDA event elapsed; allocation/clones excluded; no profiler',
                   baseline=baseline, candidates=candidates, entries=entries, skipped=skipped,
                   inventory=str(args.inventory.resolve()), complete_cartesian_search=False,
                   warmup=args.warmup, iterations=args.iterations)
    if rank == 0:
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    assert stack['depth'] == 0
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
