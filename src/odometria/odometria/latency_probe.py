"""Замер задержки «вход → публикация» и частоты выходных топиков.

Задержка каждого выходного сообщения считается как разница времени его
приёма и времени приёма последнего входного сообщения перед ним (оба по
монотонным часам этого узла). Так измеряется полный путь через DDS и узел
одометрии без зависимости от шкалы времени bag.
Запуск: ros2 run odometria latency_probe
"""
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor


def stats(values):
    if not values:
        return None
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1)))]  # noqa: E731
    return {'n': len(v), 'p50_ms': round(pick(0.5), 3), 'p95_ms': round(pick(0.95), 3),
            'p99_ms': round(pick(0.99), 3), 'max_ms': round(v[-1], 3)}


class LatencyProbe(Node):
    def __init__(self):
        super().__init__('odometria_latency_probe')
        self.declare_parameter('report_period', 10.0)
        self.declare_parameter('output_file', '')
        qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.last_input = None
        self.latency = {'velocity': [], 'position': []}
        self.counts = {'velocity': 0, 'position': 0}
        self.started = None
        for topic, kind in (('/vehicle/front_bogie_velocity', VelocitySensor),
                            ('/vehicle/rear_bogie_velocity', VelocitySensor),
                            ('/vehicle/driver_position_cmd', DriverControllerCommand)):
            self.create_subscription(kind, topic, self.on_input, qos)
        self.create_subscription(VelocitySensor, '/result/velocity', lambda m: self.on_output('velocity'), qos)
        self.create_subscription(Odometry, '/result/position', lambda m: self.on_output('position'), qos)
        self.create_timer(float(self.get_parameter('report_period').value), self.report)

    def on_input(self, _msg):
        self.last_input = time.monotonic()
        if self.started is None:
            self.started = self.last_input

    def on_output(self, name):
        now = time.monotonic()
        self.counts[name] += 1
        if self.last_input is not None:
            self.latency[name].append((now - self.last_input) * 1e3)

    def summary(self):
        elapsed = time.monotonic() - self.started if self.started else 0.0
        return {name: {'rate_hz': round(self.counts[name] / elapsed, 2) if elapsed > 0 else None,
                       'latency': stats(self.latency[name])} for name in self.latency}

    def report(self):
        if self.started is not None:
            self.get_logger().info(json.dumps(self.summary(), ensure_ascii=False))

    def save(self):
        path = self.get_parameter('output_file').value
        if path:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(self.summary(), fh, ensure_ascii=False, indent=1)


def main(args=None):
    rclpy.init(args=args)
    node = LatencyProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
