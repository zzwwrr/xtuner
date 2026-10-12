"""Shared launcher for the three Qwen experiments. Default: print a CPU-only plan."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

from metrics import read_run, summarize

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
FWD_2K = '128,64,4,64,128,64,4,64,4,4,128'
FWD_16K = '128,64,4,64,128,64,4,64,4,2,128'
BWD_BASE = '16,128,64,4,256,128,64,2,32,64,64,3,128'
BWD_TUNED = '8,128,64,4,128,256,64,2,32,64,64,3,128'


def arm(name, case, backend='liger', steps=None, fwd=None, bwd=None):
    long = case == 'qwen_16k_liger_templates'
    return dict(arm=name, backend=backend, tokens=16384 if long else 2048,
                rc=0. if case == 'qwen_2k_rc0' else 1., compile=long,
                ce='chunk' if long else 'eager', fa=3 if long else 2,
                steps=steps or (50 if long else 80),
                fwd=fwd or (FWD_16K if long else FWD_2K), bwd=bwd or BWD_BASE)


def plan(case):
    if case == 'qwen_16k_liger_templates':
        return [arm(f'confirm-{i}', case, bwd=BWD_TUNED if i in (1, 2) else BWD_BASE)
                for i in range(4)]
    return [arm(f'{i}-{backend}', case, backend=backend)
            for i, backend in enumerate(('native', 'liger', 'liger', 'native'))]


def environment(spec, out, suite):
    env = os.environ.copy()
    # Remove candidate/profiling flags inherited from another experiment.
    for key in list(env):
        if key.startswith(('LIGER_', 'UNIEP_', 'XTUNER_ROUTER_DEBUG', 'DIAGNOSTIC_', 'BENCH_')):
            del env[key]
    env.update(REPRO_SPEC=json.dumps(spec), REPRO_CONFIG=str(suite / 'train_config.py'),
               REPRO_SUITE_OUTPUT=str(out.parent), BOARD_OUT=str(out), WORK_DIR=str(out / 'work'),
               XTUNER_AB_SOURCE=str(REPO), XTUNER_DETERMINISTIC='true',
               XTUNER_USE_FA3='1' if spec['fa'] == 3 else '0',
               XTUNER_LIGER_W13_SPLIT_CAT='1' if spec['backend'] == 'liger' else '0',
               XTUNER_LIGER_NOGRAD_STRIDED='0', PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               HF_HOME=str(out / 'cache/hf'), TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
               LIGER_MOE_SM90_NONRDC='1', LIGER_ROUTE_PROBE='0', LIGER_DIAG_NO_K_SPLIT='0',
               LIGER_MOE_FORCE_CONFIG=spec['fwd'], LIGER_MOE_BWD_FORCE_CONFIG=spec['bwd'])
    env['PYTHONPATH'] = os.pathsep.join((str(REPO), str(ROOT / 'common'),
                                       str(suite), env.get('PYTHONPATH', '')))
    for key, suffix in (('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'),
                        ('CUDA_CACHE_PATH', 'cuda'), ('XDG_CACHE_HOME', 'xdg')):
        env[key] = str(out / 'cache' / suffix)
    return env


def run(spec, suite, output, args, *, single=False, capture_dir=None):
    directory = output / spec['arm']
    directory.mkdir(exist_ok=False)
    (directory / 'spec.json').write_text(json.dumps(spec, indent=2))
    env = environment(spec, directory, suite)
    # A short path avoids AF_UNIX socket limits in Python's multiprocessing queues.
    temporary = Path(tempfile.mkdtemp(prefix='repro-', dir='/root/workspace' if Path('/root/workspace').is_dir() else str(output)))
    env['TMPDIR'] = str(temporary)
    entry = ROOT / 'common/train_entry.py'
    command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
               '--log-dir', str(directory / 'ranks'), '--redirects', '3', '--tee', '3', str(entry)]
    if single:
        command[-1:] = [str(suite / 'single_sweep.py'), '--capture-dir', str(capture_dir),
                       '--out', str(directory), '--inventory', str(suite / 'template_inventory.json')]
    if spec.get('capture'):
        env.update(LIGER_SWEEP_CAPTURE_DIR=str(directory / 'capture'),
                   LIGER_SWEEP_CAPTURE_STEP='1', LIGER_SWEEP_CAPTURE_LAYER='23')
    if args.check_config:
        command = [args.python, str(entry), '--check-config']
    provenance = dict(command=command, source=str(REPO),
                      supplied_allocation_start=getattr(args, "block_start_utc", None),
                      commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                      diff=subprocess.check_output(['git', '-C', str(REPO), 'diff', '--stat'], text=True),
                      environment={key: value for key, value in env.items()
                                   if key.startswith(('REPRO_', 'XTUNER_', 'LIGER_'))
                                   or key in ('QWEN3_MOE_PATH', 'ALPACA_PATH', 'WORK_DIR', 'TMPDIR', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD')})
    (directory / 'command.json').write_text(json.dumps(provenance, indent=2))
    print('RUN', directory, flush=True)
    with (directory / 'launch.log').open('w') as log:
        process = subprocess.Popen(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except BaseException:
            # This group was created by this invocation; never kill unrelated jobs.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (directory / 'exit-code.txt').write_text(str(process.returncode))
            raise
    (directory / 'exit-code.txt').write_text(str(code))
    if code != 0:
        raise RuntimeError(f'Run failed ({code}); inspect {directory / "launch.log"}')
    # Remove only this invocation's short temporary directory after normal exit.
    shutil.rmtree(temporary)
    (directory / 'temporary-cleanup.json').write_text(json.dumps(
        dict(path=str(temporary), removed=not temporary.exists(),
             process_exit=code, utc=dt.datetime.now(dt.timezone.utc).isoformat()), indent=2))
    return directory


def main(case, suite):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-config', action='store_true', help='Build configs without training')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--timeout', type=int, default=1800, help='Per-run timeout in seconds')
    parser.add_argument('--block-start-utc', help='Optional allocation start for provenance; no fixed cutoff')
    if case == 'qwen_16k_liger_templates':
        parser.add_argument('--sweep', action='store_true', help='Capture + stock single-layer sweep + screens + ABBA')
    args = parser.parse_args()
    specs = plan(case)
    if not args.execute and not args.check_config:
        print(json.dumps(dict(case=case, gpu_started=False, runs=specs,
                              sweep=getattr(args, 'sweep', False),
                              sweep_stages=['capture2', 'single-layer menu', 'up to three screens30', 'ABBA50']
                              if getattr(args, 'sweep', False) else []), indent=2))
        return
    for key in ('QWEN3_MOE_PATH', 'ALPACA_PATH'):
        if key not in os.environ or not Path(os.environ[key]).exists():
            parser.error(f'Set {key} to an existing read-only input')
    if args.execute:
        if os.environ.get('GPU_AUTHORIZATION') != 'GRANTED_BY_USER':
            parser.error('Explicit GPU authorization is required: GPU_AUTHORIZATION=GRANTED_BY_USER')
    output = (args.out or suite / 'runs' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    for protected in (Path('/root/models'), Path('/mnt/shared-storage-user'), Path('/root/codespace'),
                      Path(os.environ['QWEN3_MOE_PATH']).resolve(), Path(os.environ['ALPACA_PATH']).resolve()):
        if output == protected or output.is_relative_to(protected):
            parser.error(f'Output must not be inside a read-only input: {protected}')
    output.mkdir(parents=True, exist_ok=False)
    if args.check_config:
        unique = [specs[0], specs[1]]
        for spec in unique:
            run(spec, suite, output, args)
        return
    if getattr(args, 'sweep', False):
        capture = arm('capture', case, steps=2)
        capture['capture'] = True
        capture_dir = run(capture, suite, output, args) / 'capture'
        single = run(arm('single', case), suite, output, args, single=True, capture_dir=capture_dir)
        # The single-layer consumer has finished; preserve routing receipts and remove its tensor inputs.
        targets = []
        for rank in range(8):
            for name in (f'replay-step1-layer23-rank{rank}.pt', f'dy-step1-layer23-rank{rank}.pt'):
                path = capture_dir / name
                if path.is_symlink() or path.resolve().parent != capture_dir.resolve():
                    raise ValueError(f'Unexpected capture target: {path}')
                path.unlink()
                targets.append(str(path))
        (capture_dir / 'tensor-cleanup.json').write_text(json.dumps(
            dict(consumer=str(single), targets=targets, removed=True), indent=2))
        candidates = sorted((row for row in json.loads((single / 'summary.json').read_text())['candidates']
                             if row['gate_pass']), key=lambda row: row['total_ms'])
        pairs, seen = [], {(FWD_16K, BWD_BASE)}
        for row in candidates:
            pair = row['fwd'], row['bwd']
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
            if len(pairs) == 2:
                break
        if not pairs:
            raise RuntimeError('No distinct candidate passed the single-layer numerical screen')
        screens = [arm('screen-baseline', case, steps=30)] + [
            arm(f'screen-{i}', case, steps=30, fwd=fwd, bwd=bwd) for i, (fwd, bwd) in enumerate(pairs)]
        screened = [(spec, read_run(run(spec, suite, output, args))[0]) for spec in screens]
        candidate = max(screened[1:], key=lambda row: row[1]['effective_tgs_per_gpu'])[0]
        specs = [arm(f'confirm-{i}', case, fwd=candidate['fwd'] if i in (1, 2) else FWD_16K,
                     bwd=candidate['bwd'] if i in (1, 2) else BWD_BASE) for i in range(4)]
    paths = [run(spec, suite, output, args) for spec in specs]
    result = summarize(paths)
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    # Every consumer is done; each cache belongs to exactly one completed arm in this suite.
    caches = [path / 'cache' for path in output.iterdir() if path.is_dir() and (path / 'exit-code.txt').is_file()]
    removed = []
    for cache in caches:
        if cache.exists():
            if cache.is_symlink() or not cache.resolve().is_relative_to(output):
                raise ValueError(f'Unexpected cache target: {cache}')
            shutil.rmtree(cache)
            removed.append(str(cache))
    (output / 'cache-cleanup.json').write_text(json.dumps(dict(targets=removed, complete=True), indent=2))
    print(json.dumps({key: result[key] for key in ('A', 'B', 'throughput_ratio', 'numerical_pass')}, indent=2))
