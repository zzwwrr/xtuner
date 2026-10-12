"""CPU-only summary of eight-rank native/Liger FFN replay timing records."""

import argparse
import json
import math
from pathlib import Path
import statistics

PHASES = ("F0_ms", "Fr_ms", "backward_including_recompute_ms", "total_ms")


def read_arm(directory):
    directory = Path(directory)
    spec = json.loads((directory / "spec.json").read_text())
    rows = []
    for rank in range(8):
        row = json.loads((directory / f"rank{rank}.json").read_text())
        gate = json.loads((directory / f"gate-rank{rank}.json").read_text())
        if row["rank"] != rank or row["spec"] != spec or not gate["passed"]:
            raise ValueError(f"Rank/spec/numerical gate mismatch: {directory}, rank {rank}")
        expected = {"Y", "dX", "dprob", "dW13", "dW2"}
        if set(gate["fp32"]) != expected:
            raise ValueError("Missing output or gradient check")
        if spec["backend"] == "liger" and set(gate["versus_native"]) != expected:
            raise ValueError("Missing Liger/native comparison")
        for checks in (gate["fp32"], gate["versus_native"]):
            for check in checks.values():
                if not (check["passed"] and check["finite"]
                        and check["bad_elements"] == 0
                        and math.isfinite(check["relative_l2"])
                        and check["relative_l2"] <= .02):
                    raise ValueError("Numerical screening failed")
        if [x["iteration"] for x in row["records"]] != list(range(50)):
            raise ValueError("Missing or duplicate timing iteration")
        for record in row["records"]:
            if not all(math.isfinite(record[k]) and record[k] > 0 for k in PHASES):
                raise ValueError("Invalid timing value")
        rows.append(row)
    samples = {key: [max(row["records"][i][key] for row in rows)
                     for i in range(50)] for key in PHASES}
    return dict(backend=spec["backend"], tokens=spec["tokens"], path=str(directory),
                means_ms={key: statistics.fmean(values) for key, values in samples.items()},
                samples_ms=samples, numerical_screening_passed=True)


def pair(native, liger):
    if native["backend"] != "native" or liger["backend"] != "liger":
        raise ValueError("Expected native then Liger")
    if native["tokens"] != liger["tokens"]:
        raise ValueError("Different input lengths")
    n, l = native["means_ms"], liger["means_ms"]
    return dict(native=native, liger=liger,
                forward_speedup=n["F0_ms"] / l["F0_ms"],
                backward_including_replay_speedup=(n["backward_including_recompute_ms"] /
                                                   l["backward_including_recompute_ms"]),
                total_speedup=n["total_ms"] / l["total_ms"])


def summarize(output, lengths):
    output = Path(output)
    result = dict(scope="Real-input single routed FFN replay; not full-model forward or training TGS",
                  backward_includes_checkpoint_replay=True,
                  aggregation="mean_over_50_iterations(max_over_8_ranks(time))",
                  results={})
    table = ["| seq_len | 方法（native or liger） | 前向时间 | 前向加速比 | 反向时间 | 反向加速比 | 总体加速比 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for tokens in lengths:
        item = pair(read_arm(output / f"native-{tokens}"), read_arm(output / f"liger-{tokens}"))
        result["results"][str(tokens)] = item
        for backend in ("native", "liger"):
            m = item[backend]["means_ms"]
            ratios = (1., 1., 1.) if backend == "native" else (
                item["forward_speedup"], item["backward_including_replay_speedup"], item["total_speedup"])
            table.append(f"| {tokens//1024}K | {backend} | {m['F0_ms']:.3f} | {ratios[0]:.3f}× | "
                         f"{m['backward_including_recompute_ms']:.3f} | {ratios[1]:.3f}× | {ratios[2]:.3f}× |")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    text = ("时间单位ms；前向为首次F0，反向包含rc1前向重放及内部激活重建。\n\n" +
            "\n".join(table) + "\n\n总体使用合计独立计时，各阶段分别取最慢rank，不要求分项最大值之和等于合计。\n")
    (output / "summary.md").write_text(text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Completed run directory")
    parser.add_argument("--seq-lens", type=int, nargs="+", choices=(8192, 16384), default=[8192, 16384])
    args = parser.parse_args()
    print(json.dumps(summarize(args.out, args.seq_lens), indent=2))


if __name__ == "__main__":
    main()
