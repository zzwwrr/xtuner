import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serial_core import reference_forward


def main():
    parser = argparse.ArgumentParser(description='EP8 synthetic MoE correctness and separate forward/backward timing')
    parser.add_argument('--backend', choices=['serial', 'liger'], required=True)
    parser.add_argument('--tokens-per-rank', type=int, required=True)
    parser.add_argument('--hidden', type=int, required=True)
    parser.add_argument('--intermediate', type=int, required=True)
    parser.add_argument('--experts', type=int, required=True)
    parser.add_argument('--topk', type=int, required=True)
    parser.add_argument('--atol', type=float, required=True)
    parser.add_argument('--rtol', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--max-relative-l2', type=float, default=0.02)
    args = parser.parse_args()
    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    if world != 8 or args.experts % world or not 0 < args.topk <= args.experts:
        raise ValueError('Requires EP8 and a valid expert layout')
    if min(args.tokens_per_rank, args.hidden, args.intermediate) <= 0 or min(args.atol, args.rtol) < 0:
        raise ValueError('Invalid shape or tolerances')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    local_experts = args.experts // world
    def leaf(shape):
        return (torch.randn(shape, device='cuda', dtype=torch.bfloat16) * 0.02).requires_grad_()
    inputs = leaf((args.tokens_per_rank, args.hidden))
    gate = leaf((local_experts, args.intermediate, args.hidden))
    up = leaf((local_experts, args.intermediate, args.hidden))
    down = leaf((local_experts, args.hidden, args.intermediate))
    probabilities, indices = torch.randn(args.tokens_per_rank, args.experts, device='cuda').softmax(-1).topk(args.topk)
    probabilities = probabilities.detach().to(torch.bfloat16).requires_grad_()
    indices = indices.to(torch.int32).contiguous()
    gradient = torch.randn_like(inputs)
    cleanup = None
    try:
        if args.backend == 'serial':
            import deep_ep
            from serial_te import load_grouped_linear
            from serial_core import serial_forward
            GroupedLinear = load_grouped_linear()
            te_up = GroupedLinear(local_experts, args.hidden, 2 * args.intermediate, bias=False, params_dtype=torch.bfloat16)
            te_down = GroupedLinear(local_experts, args.intermediate, args.hidden, bias=False, params_dtype=torch.bfloat16)
            with torch.no_grad():
                for e in range(local_experts):
                    getattr(te_up, f'weight{e}').copy_(torch.cat((gate[e], up[e])))
                    getattr(te_down, f'weight{e}').copy_(down[e])
            deep_ep.Buffer.set_num_sms(24)
            configs = [deep_ep.Buffer.get_dispatch_config(world), deep_ep.Buffer.get_combine_config(world)]
            buffer = deep_ep.Buffer(dist.group.WORLD, max(c.get_nvl_buffer_size_hint(args.hidden * 2, world) for c in configs), 0, explicitly_destroy=True)
            cleanup = buffer.destroy
            def forward():
                return serial_forward(inputs, probabilities.float(), indices.to(deep_ep.topk_idx_t), te_up, te_down, buffer, args.experts)
        elif args.backend == 'liger':
            from liger_cute_kernels import nvshmem, tvm_ffi
            from liger_kernel.ops.cute.ops import moe_fused
            nvshmem.init_from_pg()
            cleanup = nvshmem.finalize
            tvm_ffi.moe_configure_symmetric(max_tokens=args.tokens_per_rank, hidden_dim=args.hidden,
                                          max_num_experts=args.experts, max_top_k=args.topk,
                                          num_pes=world, num_hosts=1, gpus_per_host=world)
            nvshmem.resolve_team(dist.group.WORLD)
            def forward():
                return moe_fused(inputs, indices, probabilities, gate, up, down,
                                 num_experts=args.experts, top_k=args.topk, pg=dist.group.WORLD)
        output = forward()
        output.backward(gradient)
        if args.backend == 'serial':
            gate.grad = torch.stack([getattr(te_up, f'weight{e}').grad[:args.intermediate] for e in range(local_experts)])
            up.grad = torch.stack([getattr(te_up, f'weight{e}').grad[args.intermediate:] for e in range(local_experts)])
            down.grad = torch.stack([getattr(te_down, f'weight{e}').grad for e in range(local_experts)])
        ref_inputs = inputs.detach().float().requires_grad_()
        ref_probabilities = probabilities.detach().float().requires_grad_()
        ref_up = [torch.cat((gate[expert], up[expert])).detach().float().requires_grad_() for expert in range(local_experts)]
        ref_down = [down[expert].detach().float().requires_grad_() for expert in range(local_experts)]
        reference = reference_forward(ref_inputs, ref_probabilities, indices, ref_up, ref_down, dist.group.WORLD)
        reference.backward(gradient.float())
        pairs = [('Y', output, reference), ('dX', inputs.grad, ref_inputs.grad), ('dprob', probabilities.grad, ref_probabilities.grad)]
        for expert in range(local_experts):
            pairs.extend([(f'dgate{expert}', None if gate.grad is None else gate.grad[expert], ref_up[expert].grad[:args.intermediate]),
                          (f'dup{expert}', None if up.grad is None else up.grad[expert], ref_up[expert].grad[args.intermediate:]),
                          (f'ddown{expert}', None if down.grad is None else down.grad[expert], ref_down[expert].grad)])
        results = {}
        for name, actual, expected in pairs:
            finite = actual is not None and expected is not None and bool(torch.isfinite(actual).all() & torch.isfinite(expected).all())
            if finite:
                actual, expected = actual.detach().float(), expected.detach().float()
                relative_l2 = float((actual - expected).norm() / expected.norm().clamp_min(1e-30))
                results[name] = dict(finite=True, relative_l2=relative_l2,
                                     max_abs=float((actual - expected).abs().max()),
                                     passed=torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
                                     and relative_l2 <= args.max_relative_l2)
            else:
                results[name] = dict(finite=False, passed=False)

        passed = torch.tensor(int(all(item['passed'] for item in results.values())), device='cuda')
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        print(json.dumps(dict(rank=rank, config=vars(args), global_tokens=world * args.tokens_per_rank,
                              results=results, passed=bool(passed.item()), scope='tolerance-based local gate, not bitwise'), sort_keys=True), flush=True)
        if not passed.item():
            raise RuntimeError('Native MoE reference gate failed')
        # Reference is out of the measured path. Same generated inputs and gradient for all arms.
        del pairs, reference, ref_inputs, ref_probabilities, ref_up, ref_down
        timings = []
        for step in range(args.warmup + args.iters):
            for tensor in (inputs, probabilities, gate, up, down):
                tensor.grad = None
            if args.backend == 'serial':
                te_up.zero_grad(set_to_none=True)
                te_down.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize()
            start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            start.record()
            output = forward()
            middle.record()
            output.backward(gradient)
            end.record()
            end.synchronize()
            if step >= args.warmup:
                timings.append([start.elapsed_time(middle), middle.elapsed_time(end)])
        samples = torch.tensor(timings, device='cuda')
        dist.all_reduce(samples, op=dist.ReduceOp.MAX)
        print(json.dumps(dict(kind='timing', rank=rank, config=vars(args),
                              rank_local_samples_ms=timings, rank_max_samples_ms=samples.tolist(),
                              mean_fwd_ms=samples[:, 0].mean().item(), mean_bwd_ms=samples[:, 1].mean().item())), flush=True)
    finally:
        if cleanup is not None:
            cleanup()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()