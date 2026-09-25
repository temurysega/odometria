"""Узел ROS 2 Humble: резервная одометрия трамвая по модели.

Вход:  /vehicle/front_bogie_velocity, /vehicle/rear_bogie_velocity,
       /vehicle/driver_position_cmd; GNSS fix только в окне начальной
       выставки, после неё подписки на GNSS уничтожаются.
Положение: плоские координаты MGRS и высота для base_link (tf антенн в
параметрах master_x, rover_x, antenna_z).
Выход: /result/velocity (tram_vehicle_msgs/VelocitySensor),
       /result/position (nav_msgs/Odometry),
       /result/acceleration (geometry_msgs/AccelStamped),
       /result/slip_detected (std_msgs/Bool),
       /result/diagnostics (diagnostic_msgs/DiagnosticArray).
"""
from collections import deque
import math
import time

from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import AccelStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from .core import CoreConfig, OdometryCore
from .observer import ObserverConfig
from .track import TrackMap
from .traction import TractionModel

CORE_PARAMS = {
    'output_rate': 50.0, 'use_map': True, 'use_landmarks': True, 'particles': 1500,
    'init_window': 1.0, 'scale_prior': 1.0, 'scale_sigma': 0.008,
    'master_x': -9.873, 'rover_x': 2.563, 'antenna_z': 3.0, 'output_point': 'base_link',
    'mgrs_zone': 37, 'mgrs_origin_east': 300000.0, 'mgrs_origin_north': 6100000.0,
    'gnss_wait': 5.0, 'max_map_offset': 15.0, 'stop_speed': 0.03,
    'stop_confirm': 1.0, 'command_timeout': 0.5,
}
OBSERVER_PARAMS = {
    'wheel_scale': 1.0 / 3.6, 'sigma_wheel': 0.03, 'sigma_model': 0.35, 'sigma_bias': 0.02,
    'pair_abs': 0.25, 'pair_tight': 0.12, 'gate_min': 0.35, 'rise_margin': 1.6,
    'max_decel': 6.0, 'resync_after': 4.0, 'sigma_disagree': 0.2,
}
LEVEL = {'норма': DiagnosticStatus.OK, 'проскальзывание': DiagnosticStatus.WARN,
         'только модель': DiagnosticStatus.WARN, 'нет колёс': DiagnosticStatus.ERROR}


def to_time(stamp):
    ns = int(round(stamp * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometria')
        for name, value in {**CORE_PARAMS, **OBSERVER_PARAMS}.items():
            self.declare_parameter(name, value)
        self.declare_parameter('map_file', '')
        self.declare_parameter('model_file', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('diagnostics_rate', 5.0)
        core_cfg = CoreConfig(**{k: type(v)(self.get_parameter(k).value) for k, v in CORE_PARAMS.items()})
        obs_cfg = ObserverConfig(**{k: float(self.get_parameter(k).value) for k in OBSERVER_PARAMS})
        map_file = self.get_parameter('map_file').value or None
        model_file = self.get_parameter('model_file').value or None
        track_map = TrackMap.load(map_file) if core_cfg.use_map else None
        self.core = OdometryCore(core_cfg, obs_cfg, track_map=track_map, model=TractionModel.load(model_file))
        self.frame_id = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.diag_period = 1.0 / max(0.1, float(self.get_parameter('diagnostics_rate').value))
        self.last_diag = 0.0
        self.last_slip = None
        self.last_input = None
        self.timings = deque(maxlen=2000)

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(VelocitySensor, '/vehicle/front_bogie_velocity',
                                 lambda m: self.on_wheel('front', m), qos)
        self.create_subscription(VelocitySensor, '/vehicle/rear_bogie_velocity',
                                 lambda m: self.on_wheel('rear', m), qos)
        self.create_subscription(DriverControllerCommand, '/vehicle/driver_position_cmd',
                                 self.on_command, qos)
        self.gnss_subs = []
        self.qos = qos
        self.velocity_pub = self.create_publisher(VelocitySensor, '/result/velocity', 50)
        self.position_pub = self.create_publisher(Odometry, '/result/position', 50)
        self.accel_pub = self.create_publisher(AccelStamped, '/result/acceleration', 50)
        self.slip_pub = self.create_publisher(Bool, '/result/slip_detected', 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/result/diagnostics', 10)
        self.create_timer(1.0, self.watchdog)
        self._sync_gnss()
        self.get_logger().info(
            f'одометрия запущена: карта {"есть" if track_map else "нет"}, '
            f'выход {core_cfg.output_rate:.0f} Гц в кадре {self.frame_id}')

    def _sync_gnss(self):
        """GNSS нужен только до начальной выставки."""
        if self.core.gnss_needed and not self.gnss_subs:
            self.gnss_subs = [
                self.create_subscription(NavSatFix, '/sensing/gnss/master/fix',
                                         lambda m: self.on_fix('master', m), self.qos),
                self.create_subscription(NavSatFix, '/sensing/gnss/rover/fix',
                                         lambda m: self.on_fix('rover', m), self.qos)]
        elif not self.core.gnss_needed and self.gnss_subs:
            for sub in self.gnss_subs:
                self.destroy_subscription(sub)
            self.gnss_subs = []
            loc = self.core.localizer
            self.get_logger().info(
                f'начальная выставка завершена: антенна {loc.source}, '
                f'привязка к карте {"есть" if loc.start_s is not None else "нет"}; GNSS отключён')

    def on_fix(self, source, msg):
        if self.core.gnss_needed:
            self.core.on_fix(source, seconds(msg.header.stamp), msg.latitude, msg.longitude,
                             msg.altitude, msg.status.status)

    def on_command(self, msg):
        begin = time.perf_counter()
        self.last_input = time.monotonic()
        self._publish(self.core.on_command(seconds(msg.header.stamp), int(msg.position)), begin)

    def on_wheel(self, side, msg):
        begin = time.perf_counter()
        self.last_input = time.monotonic()
        self._publish(self.core.on_wheel(side, seconds(msg.header.stamp), msg.velocity), begin)

    def _publish(self, outputs, begin):
        for out in outputs:
            stamp = to_time(out.stamp)
            speed = VelocitySensor()
            speed.header.stamp = stamp
            speed.header.frame_id = self.child_frame_id
            speed.velocity = float(out.velocity)
            self.velocity_pub.publish(speed)
            if out.position_valid:
                self.position_pub.publish(self._odometry(out, stamp))
            accel = AccelStamped()
            accel.header.stamp = stamp
            accel.header.frame_id = self.child_frame_id
            accel.accel.linear.x = float(out.acceleration)
            self.accel_pub.publish(accel)
            if out.slip != self.last_slip:
                self.slip_pub.publish(Bool(data=out.slip))
                self.last_slip = out.slip
        self._sync_gnss()
        self.timings.append((time.perf_counter() - begin) * 1e3)
        if outputs and time.monotonic() - self.last_diag > self.diag_period:
            self.last_diag = time.monotonic()
            self._diagnostics(to_time(outputs[-1].stamp), outputs[-1])

    def _odometry(self, out, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        x, y, z = out.position
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = float(z)
        msg.pose.pose.orientation.z = math.sin(0.5 * out.yaw)
        msg.pose.pose.orientation.w = math.cos(0.5 * out.yaw)
        # ковариация: вдоль пути из фильтра, поперёк из точности карты
        c, s = math.cos(out.yaw), math.sin(out.yaw)
        a, b = out.var_along, out.var_cross
        cov = [0.0] * 36
        cov[0] = a * c * c + b * s * s
        cov[1] = cov[6] = (a - b) * c * s
        cov[7] = a * s * s + b * c * c
        cov[14] = 0.25 + b
        cov[21] = cov[28] = 0.01
        cov[35] = 0.003 if b < 1.0 else 1.0
        msg.pose.covariance = cov
        msg.twist.twist.linear.x = float(out.velocity)
        msg.twist.twist.angular.z = float(out.yaw_rate)
        tw = [0.0] * 36
        tw[0] = out.var_velocity
        tw[7] = tw[14] = 1e-4
        tw[21] = tw[28] = 1e-4
        tw[35] = 1e-3
        msg.twist.covariance = tw
        return msg

    def _diagnostics(self, stamp, out):
        info = self.core.diagnostics()
        times = sorted(self.timings)
        info['processing_ms_p50'] = round(times[len(times) // 2], 3) if times else None
        info['processing_ms_p99'] = round(times[int(0.99 * (len(times) - 1))], 3) if times else None
        info['processing_ms_max'] = round(times[-1], 3) if times else None
        array = DiagnosticArray()
        array.header.stamp = stamp
        status = DiagnosticStatus(level=LEVEL.get(out.status, DiagnosticStatus.WARN), name='odometria',
                                  message=out.status, hardware_id='tram_odometry')
        status.values = [KeyValue(key=k, value=str(v)) for k, v in info.items()]
        array.status = [status]
        self.diag_pub.publish(array)
        self.slip_pub.publish(Bool(data=out.slip))

    def watchdog(self):
        if self.last_input is not None and time.monotonic() - self.last_input > 2.0:
            array = DiagnosticArray()
            array.header.stamp = self.get_clock().now().to_msg()
            array.status = [DiagnosticStatus(level=DiagnosticStatus.ERROR, name='odometria',
                                             message='нет входных данных', hardware_id='tram_odometry')]
            self.diag_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # штатная остановка по сигналу
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
