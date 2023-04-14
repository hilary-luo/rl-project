from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node

def generate_launch_description():

    pkg_robot_sim = get_package_share_directory('robot_sim')
    rviz2_config = PathJoinSubstitution([pkg_robot_sim, 'rviz', 'robot.rviz'])

    return LaunchDescription([
        Node(package='rviz2',
             namespace='',
             executable='rviz2',
             name='rviz2',
             arguments=['-d', rviz2_config],
             parameters=[{'use_sim_time': False}],
             output='screen'),
        
        Node(
            package='robot_sim',
            namespace='',
            executable='robot_sim_server',
            name='robot_sim_server'
        ),  
    ])