"""Run the repository trainer while recording the actual config and dependency versions."""
import importlib.metadata
from itertools import islice
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    spec = json.loads(os.environ['REPRO_SPEC'])
    out = Path(os.environ['BOARD_OUT'])
    config = Path(os.environ['REPRO_CONFIG'])
    rank = int(os.environ.get('RANK', '0'))
    import torch
    import xtuner
    from xtuner.v1.ops.flash_attn import flash_attn_varlen_func
    from xtuner.v1.utils import XTUNER_DETERMINISTIC
    repo = Path(__file__).resolve().parents[3]
    assert Path(xtuner.__file__).resolve().is_relative_to(repo), xtuner.__file__
    cfg = runpy.run_path(str(config))['trainer']
    assert cfg.model_cfg.num_hidden_layers == 48 and cfg.model_cfg.float8_cfg is None
    assert cfg.global_batch_size == 32 and cfg.intra_layer_micro_batch == 1
    assert cfg.model_cfg.ep_size == cfg.fsdp_cfg.ep_size == 8
    assert cfg.fsdp_cfg.recompute_ratio == spec['rc']
    assert cfg.fsdp_cfg.torch_compile == cfg.model_cfg.compile_cfg == spec['compile']
    assert cfg.loss_cfg.mode == spec['ce'] and cfg.total_step == 80
    assert cfg.dataloader_cfg.pack_max_length == spec['tokens']
    assert bool(XTUNER_DETERMINISTIC)
    # CPU-only imports select the CPU attention implementation; GPU runs must use the selected FA version.
    if '--check-config' not in sys.argv:
        expected = 'gpu_flash_varlen_attn_v3' if spec['fa'] == 3 else 'flash_attn_varlen_func_v2'
        assert flash_attn_varlen_func.__name__ == expected, flash_attn_varlen_func.__name__
    versions = {}
    for name in ('torch', 'triton', 'deep-ep', 'liger-kernel', 'liger-cute-kernels',
                 'nvidia-cudnn-cu12', 'nvidia-cudnn-frontend'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    runtime = dict(python=sys.executable, xtuner=xtuner.__file__, versions=versions,
                   attention=flash_attn_varlen_func.__name__, spec=spec,
                   templates={key: os.environ.get(key) for key in (
                       'LIGER_MOE_FORCE_CONFIG', 'LIGER_MOE_BWD_FORCE_CONFIG',
                       'LIGER_MOE_SM90_NONRDC', 'XTUNER_LIGER_W13_SPLIT_CAT',
                       'XTUNER_LIGER_NOGRAD_STRIDED', 'LIGER_DIAG_NO_K_SPLIT')})
    out.mkdir(parents=True, exist_ok=True)
    (out / f'runtime-rank{rank}.json').write_text(json.dumps(runtime, indent=2))
    if rank == 0:
        (out / 'resolved-config.json').write_text(json.dumps(
            cfg.model_dump(mode='json', serialize_as_any=True), indent=2))
    if '--check-config' in sys.argv:
        print('CONFIG_OK', spec['arm'])
        return
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    from xtuner.v1.train.trainer import Trainer
    original = Trainer._data_iter
    Trainer._data_iter = lambda self: islice(original(self), spec['steps'])
    if spec.get('capture'):
        sys.path.insert(0, str(config.parent))
        from capture_hook import install
        install()
    sys.argv = ['sft', '--config', str(config)]
    runpy.run_module('xtuner.v1.train.cli.sft', run_name='__main__')


if __name__ == '__main__':
    main()
