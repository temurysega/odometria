"""Чтение rosbag2 SQLite без ROS и кэширование в npz.

Каждый топик превращается в массив строк:
  [время записи, header.stamp, значения...]
Для VelocitySensor значение одно, для DriverControllerCommand одно,
для TwistStamped шесть компонент, для NavSatFix
lat, lon, alt, status, cov_e, cov_n, cov_u, cov_type.
"""
from pathlib import Path
import sqlite3
import struct

import numpy as np

TOPICS = {
    'front': '/vehicle/front_bogie_velocity',
    'rear': '/vehicle/rear_bogie_velocity',
    'cmd': '/vehicle/driver_position_cmd',
    'master_fix': '/sensing/gnss/master/fix',
    'master_vel': '/sensing/gnss/master/vel',
    'rover_fix': '/sensing/gnss/rover/fix',
    'rover_vel': '/sensing/gnss/rover/vel',
    'out_velocity': '/result/velocity',
    'out_position': '/result/position',
}
KEY_BY_TOPIC = {v: k for k, v in TOPICS.items()}


def _header(blob):
    if blob[:4] != b'\x00\x01\x00\x00':
        raise ValueError('ожидается little endian CDR')
    sec, nsec, length = struct.unpack_from('<iII', blob, 4)
    return sec + nsec * 1e-9, 16 + length


def _aligned(offset, size):
    return 4 + ((offset - 4 + size - 1) // size) * size


def decode(blob, kind):
    stamp, off = _header(blob)
    if kind.endswith('VelocitySensor'):
        return stamp, struct.unpack_from('<d', blob, _aligned(off, 8))
    if kind.endswith('DriverControllerCommand'):
        return stamp, struct.unpack_from('<b', blob, off)
    if kind.endswith('TwistStamped'):
        return stamp, struct.unpack_from('<6d', blob, _aligned(off, 8))
    if kind.endswith('NavSatFix'):
        status, _service = struct.unpack_from('<bxH', blob, off)
        o = _aligned(off + 4, 8)
        lla = struct.unpack_from('<3d', blob, o)
        cov = struct.unpack_from('<9d', blob, o + 24)
        ctype = struct.unpack_from('<B', blob, o + 96)[0]
        return stamp, (*lla, status, cov[0], cov[4], cov[8], ctype)
    if kind.endswith('Odometry'):
        o = _aligned(off, 4)
        length = struct.unpack_from('<I', blob, o)[0]
        o = _aligned(o + 4 + length, 8)
        position = struct.unpack_from('<3d', blob, o)
        twist = struct.unpack_from('<d', blob, o + 8 * (3 + 4 + 36))[0]
        return stamp, (*position, twist)
    raise ValueError(kind)


def read_bag(folder):
    """Возвращает словарь: ключ топика и массив строк в порядке записи."""
    folder = Path(folder)
    db = next(folder.glob('*.db3'))
    con = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    topics = {i: (n, t) for i, n, t in con.execute('SELECT id, name, type FROM topics')}
    rows = {}
    for rec, tid, blob in con.execute(
            'SELECT timestamp, topic_id, data FROM messages ORDER BY timestamp'):
        name, kind = topics[tid]
        key = KEY_BY_TOPIC.get(name)
        if key is None:
            continue
        stamp, values = decode(blob, kind)
        rows.setdefault(key, []).append((rec * 1e-9, stamp, *values))
    con.close()
    return {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}


def load(folder, cache_dir=None):
    folder = Path(folder)
    if cache_dir is not None:
        cache = Path(cache_dir) / f'{folder.name}.npz'
        if cache.exists():
            with np.load(cache) as data:
                return {k: data[k] for k in data.files}
    data = read_bag(folder)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez(cache, **data)
    return data


def bag_folders(root):
    return sorted(p.parent for p in Path(root).glob('*/metadata.yaml'))


def unique_bags(root, cache_dir=None):
    """Отбрасывает побайтные копии записей (в датасете 25 таких пар)."""
    seen = {}
    out = []
    for folder in bag_folders(root):
        data = load(folder, cache_dir)
        cmd = data.get('cmd')
        key = (len(cmd), round(float(cmd[0, 0]), 6)) if cmd is not None and len(cmd) else folder.name
        if key in seen:
            continue
        seen[key] = folder.name
        out.append(folder)
    return out
