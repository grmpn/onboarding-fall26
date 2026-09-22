"""Bring up the whole sim2sim stack: simulator + policy (+ optional teleop).

    ros2 launch pup_bringup sim2sim.launch.py headless:=false teleop:=true
    ros2 launch pup_bringup sim2sim.launch.py policy_path:=/ws/runs/colab/policy.npz

`pup_sim/launch/sim_only.launch.py` is the template this is built from.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return the launch description for the full sim2sim stack."""
    # ===== TODO(student): Declare the launch arguments and the three nodes =====

    headless = LaunchConfiguration('headless')
    realtime_factor = LaunchConfiguration('realtime_factor')
    policy_path = LaunchConfiguration('policy_path')
    teleop = LaunchConfiguration('teleop')

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value="false"),
        DeclareLaunchArgument('realtime_factor', default_value="1.0"),
        DeclareLaunchArgument('teleop', default_value='false'),
        DeclareLaunchArgument('policy_path', default_value=''),

        # Simulator
        Node(
            package='pup_sim',
            executable='sim_node',
            name='pup_sim',
            output='screen',
            parameters=[{"headless": headless, "realtime_factor": realtime_factor}],
        ),

        # Policy
        Node(
            package='pup_bringup',
            executable='policy_node',
            name='pup_policy',
            output='screen',
            parameters=[{"policy_path": policy_path}],
        ),

        # Teleop
        Node(
            package='pup_sim',
            executable='teleop_node',
            name='pup_teleop',
            output='screen',
            condition=IfCondition(teleop),
        ),
    ])
    # ===== end TODO =====
