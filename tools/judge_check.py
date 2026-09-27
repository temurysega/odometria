"""Офлайн эмуляция проверочного скрипта организаторов на bag с эталоном.

Проверочный узел организаторов (check-code, hackathon_solution_checker)
сравнивает /result/velocity и /result/position с /localization/kinematic_state
через message_filters.ApproximateTimeSynchronizer с допуском 0,05 с: пары
набираются жадно по мере прихода сообщений, каждое сообщение участвует не
более одного раза. Здесь тот же алгоритм применяется к выходу ядра узла;
время прихода выхода считается равным времени записи входа, который его
породил, эталон приходит по времени записи.
Запуск: python tools/judge_check.py BAG_DIR
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'odometria'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bagdata  # noqa: E402

bagdata.TOPICS['reference'] = '/localization/kinematic_state'
bagdata.KEY_BY_TOPIC = {v: k for k, v in bagdata.TOPICS.items()}

from odometria.core import CoreConfig, OdometryCore  # noqa: E402
from odometria.observer import ObserverConfig  # noqa: E402
from odometria.track import TrackMap  # noqa: E402
from odometria.traction import TractionModel  # noqa: E402
from replay import stream  # noqa: E402


def synchronize(events, slop=0.05, queue_size=100):
    """Алгоритм ApproximateTimeSynchronizer для двух топиков.

    events: список (время прихода, номер топика, метка, значение).
    Возвращает пары (значение эталона, значение решения).
    """
    queues = [{}, {}]
    pairs = []
    for _, topic, stamp, value in sorted(events, key=lambda e: e[0]):
        mine = queues[topic]
        mine[stamp] = value
        while len(mine) > queue_size:
            del mine[min(mine)]
        other = queues[1 - topic]
        near = sorted(((abs(s - stamp), s) for s in other if abs(s - stamp) <= slop))
        for delta, s in near:
            if delta < slop:
                a, b = (other[s], mine[stamp]) if topic == 1 else (mine[stamp], other[s])
                pairs.append((a, b))
                del other[s]
                del mine[stamp]
                break
    return pairs


def run(data, core):
    """Выход ядра с временем прихода, равным времени записи входа."""
    rows = []
    for rec, kind, name, stamp, value in stream(data):
        if kind == 0:
            outs = core.on_wheel(name, stamp, value)
        elif kind == 1:
            outs = core.on_command(stamp, value)
        else:
            core.on_fix(name, stamp, *value)
            outs = []
        for o in outs:
            rows.append((rec, o))
    return rows


def score(data, rows, slop=0.05):
    ref = data['reference']
    ref_events_v = [(r[0], 0, r[1], r[5]) for r in ref]
    ref_events_p = [(r[0], 0, r[1], tuple(r[2:5])) for r in ref]
    out_v = [(rec, 1, o.stamp, o.velocity) for rec, o in rows]
    out_p = [(rec, 1, o.stamp, tuple(o.position)) for rec, o in rows if o.position_valid]
    pv = synchronize(ref_events_v + out_v, slop)
    pp = synchronize(ref_events_p + out_p, slop)
    ev = np.array([b - a for a, b in pv])
    ep = np.array([[b[i] - a[i] for i in range(3)] for a, b in pp])
    d = np.linalg.norm(ep, axis=1)
    return {
        'velocity': {'rmse': float(np.sqrt(np.mean(ev ** 2))), 'max': float(np.max(np.abs(ev))), 'n': len(ev),
                     'bias': float(np.mean(ev))},
        'position': {'x_rmse': float(np.sqrt(np.mean(ep[:, 0] ** 2))), 'y_rmse': float(np.sqrt(np.mean(ep[:, 1] ** 2))),
                     'z_rmse': float(np.sqrt(np.mean(ep[:, 2] ** 2))), 'distance_rmse': float(np.sqrt(np.mean(d ** 2))),
                     'distance_max': float(np.max(d)), 'distance_median': float(np.median(d)), 'n': len(d)},
        'reference_messages': len(ref),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path)
    parser.add_argument('--map', type=Path)
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    over = {'core': {}, 'observer': {}}
    for item in args.set:
        key, value = item.split('=', 1)
        group, name = key.split('.', 1)
        over[group][name] = json.loads(value.lower() if value in ('True', 'False') else value)
    data = bagdata.read_bag(args.bag)
    if 'reference' not in data:
        parser.error('в bag нет /localization/kinematic_state')
    core = OdometryCore(CoreConfig(**over['core']), ObserverConfig(**over['observer']),
                        track_map=TrackMap.load(args.map), model=TractionModel.load())
    result = score(data, run(data, core))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
