"""Capture real Qwen inputs and replay native/Liger FFNs; default prints the plan."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "common"))
from run_training import BWD_TUNED, FWD_16K, environment
from summarize import summarize


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def spec(tokens, backend, capture=False):
    return dict(arm=f"{'capture' if capture else backend}-{tokens}", backend=backend,
                tokens=tokens, rc=1., compile=False, ce="chunk", fa=3, steps=2,
                fwd=FWD_16K, bwd=BWD_TUNED, capture=capture, checkpoint=True)


def idle():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    if text.strip():
        raise RuntimeError("GPU compute processes exist; run this suite on an idle allocation")


def guard_output(path):
    path = Path(path).resolve()
    inputs = [Path(os.environ[k]).resolve() for k in ("QWEN3_MOE_PATH", "ALPACA_PATH")]
    for protected in [Path("/root/models"), Path("/root/codespace"),
                      Path("/mnt/shared-storage-user"), *inputs]:
        if path == protected or path.is_relative_to(protected):
            raise ValueError(f"Output must be outside read-only inputs: {path}")
    return path


def remove_owned(target, parent):
    target = Path(target)
    resolved = target.resolve()
    if target.is_symlink() or resolved == parent.resolve() or not resolved.is_relative_to(parent.resolve()):
        raise ValueError(f"Unexpected cleanup target: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    return dict(path=str(resolved), absent=not target.exists())


def launch(item, output, args, capture_dir=None):
    directory = output / item["arm"]
    directory.mkdir(exist_ok=False)
    env = environment(item, directory, HERE)
    env["REPRO_SUITE_OUTPUT"] = str(output)
    env["XTUNER_USE_CUTLASS_GROUP_GEMM"] = "0"
    env["LIGER_DIAG_PRINT_K_SPLIT"] = "0"
    temporary = Path(tempfile.mkdtemp(prefix="ffn-", dir=args.tmp_root))
    if len(os.fsencode(str(temporary))) > 60:
        remove_owned(temporary, args.tmp_root)
        raise ValueError("Use a shorter --tmp-root to avoid multiprocessing AF_UNIX limits")
    env["TMPDIR"] = str(temporary)
    if item["capture"]:
        env.update(LIGER_SWEEP_CAPTURE_DIR=str(directory / "capture"),
                   LIGER_SWEEP_CAPTURE_STEP="1", LIGER_SWEEP_CAPTURE_LAYER="23")
    entry = HERE.parent / "common/train_entry.py"
    command = [args.python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8", str(entry)]
    if args.check_config:
        command = [args.python, str(entry), "--check-config"]
    elif capture_dir is not None:
        command[-1:] = [str(HERE / "bench.py"), "--backend", item["backend"],
                       "--tokens", str(item["tokens"]), "--capture-dir", str(capture_dir), "--out", str(directory)]
    if not args.check_config:
        idle()
    save(directory / "spec.json", item)
    save(directory / "command.json", dict(command=command, environment={k: v for k, v in env.items()
         if k.startswith(("REPRO_", "LIGER_", "XTUNER_")) or k in
         ("QWEN3_MOE_PATH", "ALPACA_PATH", "WORK_DIR", "TMPDIR", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD")}))
    record = dict(started=dt.datetime.now(dt.timezone.utc).isoformat(), command=command)
    tick = time.monotonic()
    print("START", item["arm"], flush=True)
    with (directory / "launch.log").open("w") as log:
        process = subprocess.Popen(command, env=env, cwd=REPO, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record.update(pid=process.pid, pgid=process.pid, uid=os.getuid())
        save(directory / "process.json", record)
        try:
            code = process.wait(timeout=args.timeout)
        except BaseException:
            if process.poll() is None:
                proc = Path(f"/proc/{process.pid}")
                assert os.getpgid(process.pid) == process.pid and proc.stat().st_uid == os.getuid()
                assert Path(command[-1]).name.encode() in (proc / "cmdline").read_bytes() or b"torch.distributed.run" in (proc / "cmdline").read_bytes()
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=20)
            record.update(exit_code=process.returncode, elapsed_seconds=time.monotonic()-tick)
            save(directory / "process.json", record)
            raise
    record.update(exit_code=code, elapsed_seconds=time.monotonic()-tick,
                  ended=dt.datetime.now(dt.timezone.utc).isoformat())
    save(directory / "process.json", record)
    if code != 0:
        raise RuntimeError(f"{item['arm']} exited {code}; see {directory/'launch.log'}. No automatic retry.")
    if not args.check_config:
        idle()
    save(directory / "temporary-cleanup.json", remove_owned(temporary, args.tmp_root))
    print("DONE", item["arm"], flush=True)
    return directory


def capture_manifest(directory, tokens):
    root = directory / "capture"
    files, receipts = {}, []
    for rank in range(8):
        row = json.loads((root / f"capture-rank{rank}.json").read_text())
        if not (row["status"] == "COMPLETE" and row["rank"] == rank and row["same_call"]
                and row["step"] == 1 and row["layer"] == 23
                and row["shape"] == row["gradient_shape"] == [tokens, 2048]):
            raise ValueError("Incomplete or mismatched capture")
        receipts.append(row)
        for key in ("input", "gradient"):
            path = Path(row[key])
            if path.is_symlink() or path.resolve().parent != root.resolve():
                raise ValueError(f"Unexpected capture path: {path}")
            files[str(path)] = dict(size=path.stat().st_size, sha256=sha(path))
    save(directory / "capture-manifest.json", dict(files=files, receipts=receipts))
    return root


def cleanup(output):
    receipts = []
    for directory in output.iterdir():
        if not directory.is_dir() or not (directory / "process.json").exists():
            continue
        assert json.loads((directory / "process.json").read_text())["exit_code"] == 0
        work = directory / "work"
        if work.exists():
            inventory = {}
            for path in work.rglob("*"):
                if (path.is_file() and "cache" not in path.relative_to(work).parts
                        and (path.name == "tracker.jsonl" or path.suffix in (".log", ".json", ".jsonl", ".py"))):
                    target = directory / "retained-training-evidence" / path.relative_to(work)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    assert sha(path) == sha(target)
                    inventory[str(target)] = dict(size=target.stat().st_size, sha256=sha(target))
            save(directory / "training-evidence.json", inventory)
        for target in (directory / "cache", work, directory / "capture"):
            if target.exists():
                receipts.append(remove_owned(target, output))
    save(output / "cleanup.json", dict(targets=receipts, complete=True,
                                       utc=dt.datetime.now(dt.timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--check-config", action="store_true", help="Build configs only; no training")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seq-lens", nargs="+", type=int, choices=(8192, 16384), default=[8192, 16384])
    parser.add_argument("--timeout", type=int, default=900, help="Per-process timeout; failure stops the suite")
    parser.add_argument("--tmp-root", type=Path, default=Path("/root/workspace") if Path("/root/workspace").is_dir() else Path.cwd())
    args = parser.parse_args()
    if len(set(args.seq_lens)) != len(args.seq_lens):
        parser.error("Do not repeat --seq-lens values")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    specs = [item for t in args.seq_lens for item in (spec(t, "liger", True), spec(t, "native"), spec(t, "liger"))]
    if not (args.execute or args.check_config):
        print(json.dumps(dict(gpu_started=False, source=str(REPO), runs=specs,
                              warmup=10, iterations=50, ranks=8), indent=2))
        return
    for key in ("QWEN3_MOE_PATH", "ALPACA_PATH"):
        if key not in os.environ or not Path(os.environ[key]).is_dir():
            parser.error(f"Set {key} to an existing read-only input directory")
    if args.execute and os.environ.get("GPU_AUTHORIZATION") != "GRANTED_BY_USER":
        parser.error("Explicit GPU authorization required: GPU_AUTHORIZATION=GRANTED_BY_USER")
    output = guard_output(args.out or Path.cwd() / "megakernel-repro-runs" /
                          ("qwen-ffn-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    args.tmp_root = guard_output(args.tmp_root)
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "source.json", dict(root=str(REPO),
         commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
         status=subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True),
         started=dt.datetime.now(dt.timezone.utc).isoformat(), execution_requested=args.execute))
    (output / "source.diff").write_bytes(subprocess.check_output(["git", "-C", str(REPO), "diff", "HEAD", "--"]))
    if args.check_config:
        for t in args.seq_lens:
            for backend in ("native", "liger"):
                item = spec(t, backend)
                launch(item, output, args)
        cleanup(output)
        return
    idle()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True)
    if len(gpu.strip().splitlines()) != 8:
        raise RuntimeError("This reproduction requires an allocation with eight visible GPUs")
    if shutil.disk_usage(output).free < 25 * 1024**3:
        raise RuntimeError("At least 25GiB of output/scratch space is required")
    (output / "gpus.txt").write_text(gpu)
    (output / "topology.txt").write_text(subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True))
    try:
        for t in args.seq_lens:
            directory = launch(spec(t, "liger", True), output, args)
            captured = capture_manifest(directory, t)
            for backend in ("native", "liger"):
                launch(spec(t, backend), output, args, captured)
        summarize(output, args.seq_lens)
        idle()
        cleanup(output)
        save(output / "status.json", dict(status="COMPLETED", gpu_started=True))
    except BaseException as error:
        save(output / "status.json", dict(status="FAILED_STOPPED", error=repr(error),
                                          retained_for="Caller failure diagnosis; no implicit rerun"))
        raise
    print((output / "summary.md").read_text())


if __name__ == "__main__":
    main()
