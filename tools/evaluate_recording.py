"""Оценка записи, сделанной во время работы узла в ROS 2.

Запись должна содержать /result/velocity, /result/position и эталонные
топики GNSS, например:
  ros2 bag record -o run /result/velocity /result/position \
      /sensing/gnss/master/fix /sensing/gnss/master/vel /sensing/gnss/rover/fix
Метрики те же, что в tools/replay.py.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from bagdata import read_bag
from replay import metrics, reference


@dataclass
class Row:
    stamp: float
    velocity: float
    position: tuple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', type=Path)
    args = parser.parse_args()
    data = read_bag(args.recording)
    pos = {round(r[1], 3): tuple(r[2:5]) for r in data['out_position']}
    rows = [Row(r[1], r[2], pos[round(r[1], 3)]) for r in data['out_velocity'] if round(r[1], 3) in pos]
    ref = reference(data)
    duration = data['out_velocity'][-1, 0] - data['out_velocity'][0, 0]
    report = {'outputs': len(rows), 'rate_hz': round(len(rows) / duration, 2)}
    for scheme in ('out2ref', 'ref2out'):
        report[scheme] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics(rows, ref, scheme).items()
                          if not k.endswith('_sum')}
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
