"""ROS 2 Humble adapter for the longitudinal observer."""
import math
import time

from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from .estimator import Config, Observer
from .route import EARTH_RADIUS, Projector, load_routes


def seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometria')
        self.declare_parameter('traction', 1.2)
        self.declare_parameter('braking', 1.6)
        self.declare_parameter('wheel_agreement', 0.65)
        self.declare_parameter('wheel_speed_scale', 1.0 / 3.6)
        self.declare_parameter('initial_distance', 0.0)
        self.declare_parameter('use_static_route', True)
        cfg = Config(
            traction=float(self.get_parameter('traction').value),
            braking=float(self.get_parameter('braking').value),
            bogie_agreement=float(self.get_parameter('wheel_agreement').value),
            wheel_speed_scale=float(self.get_parameter('wheel_speed_scale').value),
        )
        self.observer = Observer(cfg)
        self.initial_distance = float(self.get_parameter('initial_distance').value)
        self.observer.distance = self.initial_distance
        self.routes = (load_routes() if self.get_parameter('use_static_route').value
                       else [])
        self.projector = Projector(self.routes)
        self.initial_fixes = {}
        self.first_fix_received_at = None
        self.anchor_distance = 0.0
        self.inputs = {'front': None, 'rear': None, 'command': None}
        self.started_at = None
        self.front_stamp = None
        self.front_received_at = None
        input_qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(VelocitySensor, '/vehicle/front_bogie_velocity',
                                 lambda m: self.on_wheel('front', m), input_qos)
        self.create_subscription(VelocitySensor, '/vehicle/rear_bogie_velocity',
                                 lambda m: self.on_wheel('rear', m), input_qos)
        self.create_subscription(DriverControllerCommand, '/vehicle/driver_position_cmd',
                                 self.on_command, input_qos)
        # GNSS is used at startup solely to anchor the offline route. It is
        # never passed to the velocity observer or used for later corrections.
        self.create_subscription(NavSatFix, '/sensing/gnss/master/fix',
                                 lambda m: self.on_initial_fix('master', m), input_qos)
        self.create_subscription(NavSatFix, '/sensing/gnss/rover/fix',
                                 lambda m: self.on_initial_fix('rover', m), input_qos)
        self.velocity_pub = self.create_publisher(VelocitySensor, '/result/velocity', 10)
        self.position_pub = self.create_publisher(Odometry, '/result/position', 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, '/result/diagnostics', 10)
        self.slip_pub = self.create_publisher(Bool, '/result/slip_detected', 10)

    def on_command(self, msg):
        stamp = seconds(msg.header.stamp)
        if math.isfinite(stamp) and stamp > 0:
            self.inputs['command'] = (time.monotonic(), int(msg.position))

    def on_initial_fix(self, source, msg):
        if self.projector.anchor is not None or source in self.initial_fixes:
            return
        lat, lon, alt = msg.latitude, msg.longitude, msg.altitude
        if (msg.status.status < 0 or not all(math.isfinite(v) for v in (lat, lon, alt))
                or not -90 <= lat <= 90 or not -180 <= lon <= 180):
            return
        received_at = time.monotonic()
        self.initial_fixes[source] = (lat, lon, alt)
        if self.first_fix_received_at is None:
            self.first_fix_received_at = received_at
        self.maybe_initialize(received_at)

    def maybe_initialize(self, now):
        if self.projector.anchor is not None or not self.initial_fixes:
            return
        if len(self.initial_fixes) == 1 and now - self.first_fix_received_at < 0.7:
            return
        anchor = self.initial_fixes.get('master') or self.initial_fixes['rover']
        heading = (None, None)
        if len(self.initial_fixes) == 2:
            mlat, mlon, _ = self.initial_fixes['master']
            rlat, rlon, _ = self.initial_fixes['rover']
            heading = (EARTH_RADIUS * math.radians(rlon - mlon) *
                       math.cos(math.radians(mlat)),
                       EARTH_RADIUS * math.radians(rlat - mlat))
        if self.projector.initialize(*anchor, *heading):
            self.anchor_distance = self.observer.distance - self.initial_distance

    def on_wheel(self, side, msg):
        stamp = seconds(msg.header.stamp)
        if not math.isfinite(stamp) or stamp <= 0:
            return
        arrived_at = time.monotonic()
        if side == 'front':
            if self.front_stamp is not None and stamp < self.front_stamp - 1.0:
                self.observer = Observer(self.observer.cfg)
                self.observer.distance = self.initial_distance
                self.projector = Projector(self.routes)
                self.initial_fixes = {}
                self.first_fix_received_at = None
                self.anchor_distance = 0.0
                self.started_at = None
                self.inputs = {'front': None, 'rear': None, 'command': None}
            self.front_stamp = stamp
            self.front_received_at = arrived_at
        self.inputs[side] = (arrived_at, float(msg.velocity))
        if self.started_at is None:
            self.started_at = arrived_at
        relative_time = arrived_at - self.started_at

        def current(name):
            saved = self.inputs[name]
            if saved is None or saved[0] > arrived_at:
                return None, float('inf')
            return saved[1], arrived_at - saved[0]

        command, command_age = current('command')
        front, front_age = current('front')
        rear, rear_age = current('rear')
        # Missing input is represented with a finite age beyond its timeout.
        if not math.isfinite(command_age):
            command_age = self.observer.cfg.command_timeout + 1.0
        if not math.isfinite(front_age):
            front_age = self.observer.cfg.velocity_timeout + 1.0
        if not math.isfinite(rear_age):
            rear_age = self.observer.cfg.velocity_timeout + 1.0
        estimate = self.observer.step(relative_time, command, front, rear,
                                      command_age=command_age,
                                      front_age=front_age, rear_age=rear_age)
        self.maybe_initialize(arrived_at)
        # The front header shares the time base of the master GNSS reference in
        # the supplied bags. Rear and controller headers have changing offsets.
        # Advance the most recent front stamp by elapsed receive time for rear
        # events, while retaining exact front stamps where available.
        output_time = stamp if side == 'front' or self.front_stamp is None else (
            self.front_stamp + arrived_at - self.front_received_at)
        sec = int(output_time)
        nanosec = round((output_time - sec) * 1e9)
        if nanosec == 1_000_000_000:
            sec += 1
            nanosec = 0
        output_stamp = Time(sec=sec, nanosec=nanosec)
        speed = VelocitySensor()
        speed.header.stamp = output_stamp
        speed.header.frame_id = 'base_link'
        speed.velocity = estimate.velocity
        self.velocity_pub.publish(speed)

        pose = Odometry()
        pose.header.stamp = output_stamp
        pose.header.frame_id = 'odom'
        pose.child_frame_id = 'base_link'
        east, north, up = self.projector.position(
            estimate.distance - self.initial_distance - self.anchor_distance)
        pose.pose.pose.position.x = east
        pose.pose.pose.position.y = north
        pose.pose.pose.position.z = up
        pose.pose.pose.orientation.w = 1.0
        pose.twist.twist.linear.x = estimate.velocity
        pose.pose.covariance[0] = estimate.sigma_distance ** 2
        pose.twist.covariance[0] = estimate.sigma_velocity ** 2
        # Map fit and longitudinal uncertainty bound the reported 2D position.
        # Heading and map shape remain uncertain, especially off the known route.
        map_variance = 25.0 if self.projector.start_s is not None else 1e6
        pose.pose.covariance[0] += map_variance
        pose.pose.covariance[7] = estimate.sigma_distance ** 2 + map_variance
        pose.pose.covariance[14] = map_variance
        pose.pose.covariance[35] = 1e6
        self.position_pub.publish(pose)

        slip = estimate.status in ('suspect_wheels', 'bogie_disagreement')
        self.slip_pub.publish(Bool(data=slip))
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = output_stamp
        level = (DiagnosticStatus.OK if estimate.status == 'ok' else
                 DiagnosticStatus.WARN if estimate.status in
                 ('suspect_wheels', 'bogie_disagreement') else DiagnosticStatus.ERROR)
        diagnostics.status = [DiagnosticStatus(
            level=level, name='odometria', message=estimate.status,
            hardware_id='tram_odometry', values=[
                KeyValue(key='trusted_bogies', value=str(estimate.trusted_bogies)),
                KeyValue(key='sigma_velocity_mps', value=f'{estimate.sigma_velocity:.3f}'),
                KeyValue(key='sigma_distance_m', value=f'{estimate.sigma_distance:.3f}'),
                KeyValue(key='route_error_m', value=str(self.projector.map_error)),
            ])]
        self.diagnostics_pub.publish(diagnostics)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
