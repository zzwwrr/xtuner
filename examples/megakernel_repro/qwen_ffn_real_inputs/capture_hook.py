"""Capture one real 8K or 16K Liger call and its exact upstream gradient.

Call install() immediately before the normal trainer entry. This observes the
first grad-enabled call at step/layer 1/23 (the checkpoint recomputation), so
gradient accumulation cannot pair one microbatch's X with another one's dY.
"""
import json
import os
from pathlib import Path


def install():
    import torch
    from xtuner.v1.profiler.prober import ProberList
    from xtuner.v1.module.moe_backend.liger import LigerCuteSession

    out = Path(os.environ['LIGER_SWEEP_CAPTURE_DIR']).resolve()
    approved = Path(os.environ['REPRO_SUITE_OUTPUT']).resolve()
    assert out.is_relative_to(approved), 'Capture output must belong to this task'
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    target_step = int(os.environ.get('LIGER_SWEEP_CAPTURE_STEP', '1'))
    target_layer = int(os.environ.get('LIGER_SWEEP_CAPTURE_LAYER', '23'))
    tokens = json.loads(os.environ['REPRO_SPEC'])['tokens']
    assert tokens in (8192, 16384)
    state = {'step': 0, 'saved': False}
    original_step = ProberList.set_step
    original_call = LigerCuteSession.__call__

    def set_step(value):
        state['step'] = int(value)
        return original_step(value)

    ProberList.set_step = staticmethod(set_step)

    def local(t):
        return t.to_local() if hasattr(t, 'to_local') else t

    @torch._dynamo.disable
    def call(self, hidden_states, topk_ids, topk_weights):
        selected = (not state['saved'] and state['step'] == target_step
                    and getattr(self.layer, 'layer_idx', None) == target_layer
                    and torch.is_grad_enabled())
        if selected:
            assert hidden_states.shape == (tokens, 2048), hidden_states.shape
            assert topk_ids.shape == topk_weights.shape == (tokens, 8)
            payload = dict(x=hidden_states.detach().cpu(), ids=topk_ids.detach().cpu(),
                           prob=topk_weights.detach().cpu(),
                           w13=local(self.layer.experts.fused_w1w3.weight).detach().cpu(),
                           w2=local(self.layer.experts.fused_w2.weight).detach().cpu())
            assert all(torch.isfinite(t).all() for t in payload.values())
            state['saved'] = True
            replay = out / f'replay-step{target_step}-layer{target_layer}-rank{rank}.pt'
            gradient = out / f'dy-step{target_step}-layer{target_layer}-rank{rank}.pt'
            receipt = out / f'capture-rank{rank}.json'
            assert not replay.exists() and not gradient.exists(), 'Do not overwrite capture'
            torch.save(payload, replay)
            info = dict(status='INPUT_SAVED', rank=rank, step=target_step,
                        layer=target_layer, input=str(replay), gradient=str(gradient),
                        shape=list(hidden_states.shape), same_call=True,
                        method='First gradient-enabled routed MoE call; checkpoint recomputation',
                        source=os.environ['XTUNER_AB_SOURCE'])
            receipt.write_text(json.dumps(info, indent=2))
        y = original_call(self, hidden_states, topk_ids, topk_weights)
        if selected:
            assert y.requires_grad
            def save_grad(dy):
                assert not gradient.exists(), 'One capture receives exactly one upstream gradient'
                assert dy.shape == hidden_states.shape and torch.isfinite(dy).all()
                torch.save(dy.detach().contiguous().cpu(), gradient)
                info['status'] = 'COMPLETE'
                info['gradient_shape'] = list(dy.shape)
                receipt.write_text(json.dumps(info, indent=2))
            y.register_hook(save_grad)
        return y

    LigerCuteSession.__call__ = call
    return state
