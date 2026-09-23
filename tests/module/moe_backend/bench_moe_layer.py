"""Single-layer MoE routed-expert micro-benchmark (native vs liger), multi-GPU.

Times the *whole routed-expert segment* exactly as ``MoEDecoderLayer._forward``
runs it — router/gate, shared experts, attention, residual are excluded:

  native: dispatch_preprocess -> dispatch -> dispatch_postprocess
          -> MoEBlock (grouped GEMM x2 + SwiGLU)
          -> combine_preprocess -> combine -> combine_postprocess
  liger:  one LigerCuteSession fused call (dispatch+GEMM1+SwiGLU+GEMM2+combine
          over NVSHMEM symmetric memory)

This is the layer-level A/B the full-training runs cannot isolate: same
MoEBlock parameter storage, same router-produced topk inputs, forward+backward
(activation-checkpoint recompute is NOT modelled — both arms run eager
fwd+bwd, so the comparison stays symmetric).

Launch — one backend per torchrun (like the training A/B), via the wrapper
which sets PYTHONPATH / NVSHMEM preload / caches::

    GPU_AUTHORIZATION=GRANTED_BY_USER bash run_moe_bench.sh native
    GPU_AUTHORIZATION=GRANTED_BY_USER bash run_moe_bench.sh liger

Env knobs (defaults = GLM5.2-30B single-layer shape, EP8):

    BENCH_H=6144 BENCH_I=2048 BENCH_E=256 BENCH_K=8
    BENCH_T=4096          # tokens per rank entering the layer
    BENCH_WARMUP=10 BENCH_ITERS=30
    BENCH_FWD_ONLY=0      # 1 = forward only (no backward)
    BENCH_MAX_TOKENS=""   # liger symmetric workspace size; default = BENCH_T
    BENCH_OUT_DIR=...     # where rank0 writes the JSON summary

Qwen3-30B-A3B shape: BENCH_H=2048 BENCH_I=768 BENCH_E=128 BENCH_K=8.
Requires world size in {4, 8} (EP group = whole world).
"""

from __future__ import annotations

import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

BACKEND = os.environ.get("BENCH_BACKEND", "native")  # native | liger | uniep
H = int(os.environ.get("BENCH_H", 6144))
I = int(os.environ.get("BENCH_I", 2048))
E = int(os.environ.get("BENCH_E", 256))
K = int(os.environ.get("BENCH_K", 8))
T = int(os.environ.get("BENCH_T", 4096))
WARMUP = int(os.environ.get("BENCH_WARMUP", 10))
ITERS = int(os.environ.get("BENCH_ITERS", 30))
FWD_ONLY = os.environ.get("BENCH_FWD_ONLY", "0") == "1"
MAX_TOKENS_RAW = os.environ.get("BENCH_MAX_TOKENS", "")
MAX_TOKENS = int(MAX_TOKENS_RAW) if MAX_TOKENS_RAW else T
OUT_DIR = os.environ.get("BENCH_OUT_DIR", ".")


def _log(rank: int, msg: str):
    print(f"[bench rank{rank}] {msg}", flush=True)


def _native_segment(x, dispatcher, experts, ep_mesh, topk_ids, topk_weights):
    """The native routed-expert segment extracted 1:1 from
    ``MoEDecoderLayer._forward`` (bf16, decoding=False)."""
    pre = dispatcher.dispatch_preprocess(
        hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights
    )
    dispatched = dispatcher.dispatch(
        pre_dispatched=pre, topk_weights=topk_weights, decoding=False
    )
    post = dispatcher.dispatch_postprocess(pre_dispatched=pre, dispatched=dispatched)
    if ep_mesh is not None:
        torch._dynamo.mark_dynamic(post["hidden_states"], 0)
    experts_out = experts(post["hidden_states"], post["tokens_per_expert"], decoding=False)
    pre_combined = dispatcher.combine_preprocess(
        hidden_states=experts_out,
        pre_dispatched=pre,
        dispatched=dispatched,
        post_dispatched=post,
        decoding=False,
    )
    combined = dispatcher.combine(
        pre_dispatched=pre,
        dispatched=dispatched,
        post_dispatched=post,
        pre_combined=pre_combined,
        decoding=False,
    )
    post_combined = dispatcher.combine_postprocess(
        pre_dispatched=pre,
        dispatched=dispatched,
        post_dispatched=post,
        pre_combined=pre_combined,
        combined=combined,
    )
    return post_combined["hidden_states"]


def main():  # pragma: no cover - multi-rank entry
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    torch.manual_seed(42 + rank)

    if world not in (4, 8):
        raise ValueError("bench requires 4 or 8 GPUs on one host (EP = world).")
    if E % world != 0:
        raise ValueError(f"BENCH_E={E} must be divisible by world size {world}.")

    ep_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))

    # ---- real MoEBlock for parameter storage (native layout) ------------
    from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock

    block = MoEBlock(
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
    with torch.no_grad():  # deterministic init; ranks own disjoint expert shards
        for pi, (_, p) in enumerate(block.named_parameters()):
            local = p.to_local() if hasattr(p, "to_local") else p
            gen = torch.Generator(device="cpu").manual_seed(7 + rank * 10 + pi)
            local.copy_(torch.randn(local.shape, generator=gen).to(local.device) * 0.02)
    for p in block.parameters():
        p.requires_grad_(True)

    # ---- router-shaped inputs (softmax top-k over E experts, fp32 weights)
    gen = torch.Generator(device="cpu").manual_seed(1234 + rank)
    logits = torch.randn(T, E, generator=gen).cuda().float()
    probs = torch.softmax(logits, dim=-1)
    topk_w, topk_i = torch.topk(probs, K, dim=-1)  # fp32 weights, long ids
    topk_w = topk_w.detach().requires_grad_(True)

    x0 = torch.randn(T, H, generator=gen).cuda().bfloat16() * 0.5
    out_grad = torch.randn(T, H, generator=gen).cuda().bfloat16()

    if BACKEND == "native":
        from xtuner.v1.module.dispatcher import build_dispatcher

        dispatcher = build_dispatcher(
            dispatcher="all2all",
            n_routed_experts=E,
            ep_group=group,
            tp_group=None,
            ep_tp_group=None,
            training_dtype="bf16",
            generate_dtype="bf16",
        )

        def run_seg(x):
            return _native_segment(x, dispatcher, block, ep_mesh, topk_i, topk_w)

    elif BACKEND == "liger":
        from xtuner.v1.module.moe_backend import build_fused_ep_forward

        class _LayerShim:
            experts = block
            _moe_top_k = K

        session = build_fused_ep_forward(
            "liger", layer=_LayerShim, ep_group=group,
            max_tokens=MAX_TOKENS, num_sm=64, num_hosts=1,
        )

        def run_seg(x):
            return session(x, topk_i, topk_w)

    elif BACKEND == "uniep":
        from xtuner.v1.module.moe_backend import build_fused_ep_forward

        class _LayerShim:
            experts = block
            _moe_top_k = K

        session = build_fused_ep_forward(
            "uniep", layer=_LayerShim, ep_group=group,
            max_tokens=MAX_TOKENS, num_sm=64, num_hosts=1,
        )

        def run_seg(x):
            return session(x, topk_i, topk_w)

    else:
        raise ValueError(f"BENCH_BACKEND must be native|liger|uniep, got {BACKEND!r}")

    # ---- sanity: one run, finite check ----------------------------------
    xs = x0.detach().clone().requires_grad_(not FWD_ONLY)
    with torch.no_grad() if FWD_ONLY else torch.enable_grad():
        y = run_seg(xs)
    if not FWD_ONLY:
        y.backward(out_grad)
    torch.cuda.synchronize()
    if not torch.isfinite(y.float()).all():
        raise RuntimeError(f"rank{rank}: non-finite output, aborting benchmark")
    _log(rank, f"sanity ok backend={BACKEND} y_norm={y.float().norm().item():.4f} "
               f"shape={tuple(y.shape)}")

    # ---- timed loop ------------------------------------------------------
    times_ms: list[float] = []
    for it in range(WARMUP + ITERS):
        x = x0.detach().clone().requires_grad_(not FWD_ONLY)
        topk_w.grad = None
        for p in block.parameters():
            p.grad = None
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        phase = "timed" if it >= WARMUP else "warmup"
        start.record()
        torch.cuda.nvtx.range_push(f"bench_{phase}")
        with torch.no_grad() if FWD_ONLY else torch.enable_grad():
            y = run_seg(x)
            if not FWD_ONLY:
                y.backward(out_grad)
        torch.cuda.nvtx.range_pop()
        end.record()
        torch.cuda.synchronize()
        if it >= WARMUP:
            times_ms.append(start.elapsed_time(end))

    if not FWD_ONLY:
        grads = [x.grad, topk_w.grad] + [
            p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            for p in block.parameters()
        ]
        if not all(t is not None and torch.isfinite(t).all().item() for t in grads):
            raise RuntimeError('Nonfinite or missing benchmark gradients')
    rank_max_times = torch.tensor(times_ms, device='cuda')
    dist.all_reduce(rank_max_times, op=dist.ReduceOp.MAX)
    mean_ms = statistics.fmean(times_ms)
    med_ms = statistics.median(times_ms)
    min_ms, max_ms = min(times_ms), max(times_ms)
    _log(rank, f"{BACKEND} T={T} H={H} E={E} K={K} ep={world} "
               f"mean={mean_ms:.3f}ms med={med_ms:.3f}ms "
               f"min={min_ms:.3f} max={max_ms:.3f} fwd_only={FWD_ONLY}")

    # ---- aggregate across ranks -----------------------------------------
    stats = torch.tensor([mean_ms, med_ms, min_ms, max_ms], device="cuda")
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
    stats /= world
    glob_mean_ms, glob_med_ms, glob_min_ms, glob_max_ms = stats.tolist()
    if rank == 0:
        toks_per_s = T * world / glob_mean_ms * 1000.0
        expanded_per_s = T * K * world / glob_mean_ms * 1000.0
        payload = {
            "backend": BACKEND,
            "router_gradient": not FWD_ONLY,
            "rank_max_times_ms": rank_max_times.tolist(),
            "mean_ms_rank_max": rank_max_times.mean().item(),
            "shape": {"H": H, "I": I, "E": E, "K": K, "T_per_rank": T, "ep": world},
            "warmup": WARMUP,
            "iters": ITERS,
            "fwd_only": FWD_ONLY,
            "fused_max_tokens": MAX_TOKENS if BACKEND in ("liger", "uniep") else None,
            "mean_ms_avg_over_ranks": glob_mean_ms,
            "median_ms_avg_over_ranks": glob_med_ms,
            "min_ms_avg_over_ranks": glob_min_ms,
            "max_ms_avg_over_ranks": glob_max_ms,
            "rank0_times_ms": times_ms,
            "global_tokens_per_s": toks_per_s,
            "global_expanded_tokens_per_s": expanded_per_s,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, f"moe-layer-{BACKEND}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        _log(0, f" wrote {out_path}: mean={glob_mean_ms:.3f}ms "
                 f"tokens/s={toks_per_s:.0f} expanded/s={expanded_per_s:.0f}")
    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
