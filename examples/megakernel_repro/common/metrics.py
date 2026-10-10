"""CPU-only tracker analysis using effective tokens and the slowest rank per step."""
import argparse
import json
import math
from pathlib import Path
import statistics


def read_run(directory):
    directory = Path(directory)
    spec = json.loads((directory / 'spec.json').read_text())
    if (directory / 'exit-code.txt').read_text().strip() != '0':
        raise ValueError(f'Failed run: {directory}')
    ranks = []
    for rank in range(8):
        matches = list((directory / 'work').glob(f'**/exp_tracking/rank{rank}/tracker.jsonl'))
        if len(matches) != 1:
            raise ValueError(f'Expected one tracker for rank {rank}: {matches}')
        rows = [json.loads(line) for line in matches[0].read_text().splitlines() if line.strip()]
        if [row['step'] for row in rows] != list(range(1, spec['steps'] + 1)):
            raise ValueError(f'Incomplete, duplicated or unordered steps: rank {rank}')
        for row in rows:
            for key in ('time/step_time', 'runtime_info/text_tokens', 'lr', 'grad_norm',
                        'loss/local_loss', 'loss/reduced_llm_loss',
                        'memory/max_memory_GB', 'memory/reserved_memory_GB'):
                if not math.isfinite(row[key]):
                    raise ValueError(f'Invalid {key} in rank {rank}')
            if row['time/step_time'] <= 0:
                raise ValueError('Nonpositive step time')
        ranks.append(rows)
    # Match historical windows: 11–79 for 80 steps, 11–49 for 50, 11–29 for 30.
    indices = range(10, spec['steps'] - 1)
    if spec['steps'] <= 11:
        raise ValueError('Capture/smoke runs are not throughput measurements')
    seconds = sum(max(rows[i]['time/step_time'] for rows in ranks) for i in indices)
    tokens = sum(rows[i]['runtime_info/text_tokens'] for rows in ranks for i in indices)
    result = dict(path=str(directory), spec=spec, window=[11, spec['steps'] - 1],
                  seconds=seconds, text_tokens=tokens, mean_seconds=seconds / len(indices),
                  effective_tgs_per_gpu=tokens / (8 * seconds),
                  allocated_peak_GiB=max(row['memory/max_memory_GB'] for rows in ranks for row in rows),
                  reserved_peak_GiB=max(row['memory/reserved_memory_GB'] for rows in ranks for row in rows))
    return result, ranks


def compare_rows(left, right):
    failures = []
    for rank, (xs, ys) in enumerate(zip(left, right)):
        if len(xs) != len(ys):
            raise ValueError('Different run lengths')
        for x, y in zip(xs, ys):
            for key in ('runtime_info/text_tokens', 'runtime_info/seqlen_tokens', 'lr'):
                if x[key] != y[key]:
                    raise ValueError(f'Inputs/LR differ: rank={rank}, step={x["step"]}, {key}')
            limits = {'loss/local_loss': .01, 'loss/reduced_llm_loss': .01,
                      'grad_norm': max(.01, abs(x['grad_norm']) * .05)}
            for key, limit in limits.items():
                if abs(x[key] - y[key]) > limit:
                    failures.append(dict(rank=rank, step=x['step'], metric=key,
                                         reference=x[key], candidate=y[key], limit=limit))
    return dict(passed=not failures, failures=failures, scope='Short-run diagnostic thresholds')


def summarize(paths):
    if len(paths) != 4:
        raise ValueError('Expected four complete ABBA runs')
    loaded = [read_run(path) for path in paths]
    runs = [row[0] for row in loaded]
    for key in ('tokens', 'rc', 'compile', 'ce', 'fa', 'steps'):
        if len({run['spec'][key] for run in runs}) != 1:
            raise ValueError(f'ABBA workload differs in {key}')
    for a, b in ((0, 3), (1, 2)):
        for key in ('backend', 'fwd', 'bwd'):
            if runs[a]['spec'][key] != runs[b]['spec'][key]:
                raise ValueError(f'ABBA repeat differs in {key}')
    checks = [dict(pair=[a, b], **compare_rows(loaded[a][1], loaded[b][1]))
              for a, b in ((0, 1), (3, 2), (0, 3), (1, 2))]
    def merged(indices):
        seconds = sum(runs[i]['seconds'] for i in indices)
        tokens = sum(runs[i]['text_tokens'] for i in indices)
        return dict(effective_tgs_per_gpu=tokens / (8 * seconds),
                    mean_seconds=statistics.fmean(runs[i]['mean_seconds'] for i in indices))
    a, b = merged((0, 3)), merged((1, 2))
    return dict(runs=runs, A=a, B=b, throughput_ratio=b['effective_tgs_per_gpu'] / a['effective_tgs_per_gpu'],
                step_speedup=a['mean_seconds'] / b['mean_seconds'], checks=checks,
                numerical_pass=all(check['passed'] for check in checks),
                note='Performance and numerical checks are reported separately; profile runs are excluded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs=4)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = summarize(args.paths)
    body = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(body)
    print(body)
