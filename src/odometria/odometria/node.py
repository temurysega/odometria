"""ROS 2 Humble adapter for the longitudinal observer."""
import math
import time

from builtin_interfaces.msg import Time
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from .estimator import Config, Observer


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
        cfg = Config(
            traction=float(self.get_parameter('traction').value),
            braking=float(self.get_parameter('braking').value),
            bogie_agreement=float(self.get_parameter('wheel_agreement').value),
            wheel_speed_scale=float(self.get_parameter('wheel_speed_scale').value),
        )
        self.observer = Observer(cfg)
        self.initial_distance = float(self.get_parameter('initial_distance').value)
        self.observer.distance = self.initial_distance
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
        self.velocity_pub = self.create_publisher(VelocitySensor, '/result/velocity', 10)
        self.position_pub = self.create_publisher(Odometry, '/result/position', 10)

    def on_command(self, msg):
        stamp = seconds(msg.header.stamp)
        if math.isfinite(stamp) and stamp > 0:
            self.inputs['command'] = (time.monotonic(), int(msg.position))

    def on_wheel(self, side, msg):
        stamp = seconds(msg.header.stamp)
        if not math.isfinite(stamp) or stamp <= 0:
            return
        arrived_at = time.monotonic()
        if side == 'front':
            if self.front_stamp is not None and stamp < self.front_stamp - 1.0:
                self.observer = Observer(self.observer.cfg)
                self.observer.distance = self.initial_distance
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
        pose.pose.pose.position.x = estimate.distance
        pose.pose.pose.orientation.w = 1.0
        pose.twist.twist.linear.x = estimate.velocity
        pose.pose.covariance[0] = estimate.sigma_distance ** 2
        pose.twist.covariance[0] = estimate.sigma_velocity ** 2
        # Lateral and vertical position and yaw cannot be inferred from scalar
        # wheel speeds. Advertise large uncertainty instead of false precision.
        pose.pose.covariance[7] = 1e6
        pose.pose.covariance[14] = 1e6
        pose.pose.covariance[35] = 1e6
        self.position_pub.publish(pose)


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
