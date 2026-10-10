"""Read the two completed isolated benchmark logs; never execute GPU code."""
import argparse
import json
from pathlib import Path


def timing(directory):
    directory = Path(directory)
    if (directory / 'exit-code.txt').read_text().strip() != '0':
        raise ValueError(f'Failed benchmark: {directory}')
    rows = []
    for path in (directory / 'ranks').rglob('stdout.log'):
        for line in path.read_text().splitlines():
            try:
                value = json.loads(line[line.index('{'):])
            except (ValueError, json.JSONDecodeError):
                continue
            if value.get('kind') == 'timing':
                rows.append(value)
    if sorted(row['rank'] for row in rows) != list(range(8)):
        raise ValueError('Expected exactly one timing record per rank')
    keys = ('tokens_per_rank', 'hidden', 'intermediate', 'experts', 'topk', 'seed', 'warmup', 'iters')
    for key in keys:
        if len({row['config'][key] for row in rows}) != 1:
            raise ValueError(f'Rank configs differ: {key}')
    for key in ('mean_fwd_ms', 'mean_bwd_ms'):
        if any(row[key] != rows[0][key] for row in rows):
            raise ValueError(f'Rank-max timing differs: {key}')
    row = rows[0]
    return dict(config=row['config'], forward_ms=row['mean_fwd_ms'], backward_ms=row['mean_bwd_ms'],
                total_ms=row['mean_fwd_ms'] + row['mean_bwd_ms'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('serial')
    parser.add_argument('liger')
    args = parser.parse_args()
    a, b = timing(args.serial), timing(args.liger)
    assert a['config']['backend'] == 'serial' and b['config']['backend'] == 'liger'
    assert {k: v for k, v in a['config'].items() if k != 'backend'} == {
        k: v for k, v in b['config'].items() if k != 'backend'}
    print(json.dumps(dict(serial=a, liger=b, forward_speedup=a['forward_ms'] / b['forward_ms'],
                          backward_speedup=a['backward_ms'] / b['backward_ms'],
                          total_speedup=a['total_ms'] / b['total_ms']), indent=2))
