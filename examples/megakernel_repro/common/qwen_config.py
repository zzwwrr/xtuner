"""Portable form of the historical Qwen configuration overlays."""
import json
import os
from pathlib import Path
import runpy


def build():
    from xtuner.v1.loss.ce_loss import CELossConfig

    repo = Path(__file__).resolve().parents[3]
    spec = json.loads(os.environ['REPRO_SPEC'])
    trainer = runpy.run_path(str(repo / 'autotest/config/qwen3_moe_30BA3_ep8.py'))['trainer']
    trainer.model_cfg.dispatcher = 'deepep'
    trainer.model_cfg.ep_size = trainer.fsdp_cfg.ep_size = 8
    trainer.model_cfg.moe_backend = spec['backend']
    trainer.model_cfg.moe_backend_max_tokens = spec['tokens']
    trainer.model_cfg.moe_backend_num_hosts = 1
    trainer.model_cfg.float8_cfg = None
    trainer.model_cfg.compile_cfg = trainer.fsdp_cfg.torch_compile = spec['compile']
    trainer.fsdp_cfg.recompute_ratio = spec['rc']
    trainer.fsdp_cfg.cpu_offload = False
    trainer.global_batch_size = 32
    trainer.intra_layer_micro_batch = 1
    trainer.dataloader_cfg.pack_max_length = spec['tokens']
    for entry in trainer.dataset_cfg:
        entry['tokenize_fn'].max_length = spec['tokens']
        entry['dataset'].cache_dir = str(Path(os.environ['WORK_DIR']) / 'cache/data')
    trainer.loss_cfg = CELossConfig(mode=spec['ce'], chunk_size=1024, loss_reduction='token')
    trainer.model_cfg.lm_loss_cfg = trainer.loss_cfg
    # Short capture/screens/confirmations stop the iterator, retaining the original LR schedule.
    trainer.total_step = 80
    trainer.total_epoch = None
    trainer.seed = 0
    trainer.optim_cfg.foreach = None
    trainer.load_from = trainer.tokenizer_path = os.environ['QWEN3_MOE_PATH']
    trainer.work_dir = os.environ['WORK_DIR']
    trainer.checkpoint_interval = trainer.hf_interval = trainer.checkpoint_maxkeep = None
    trainer.debug_skip_save = True
    trainer.profile_time = True
    return trainer
