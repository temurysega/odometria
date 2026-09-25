"""Запуск узла одометрии с параметрами из config/odometria.yaml."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default = str(Path(get_package_share_directory('odometria')) / 'config' / 'odometria.yaml')
    params = LaunchConfiguration('params')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default),
        Node(package='odometria', executable='odometry_node', name='odometria',
             output='screen', parameters=[params]),
    ])
