# Interface to convert clicked points on RViz into custom sim nav commands
# ros2 run cas726_project human_explorer_sim

import numpy as np
import rclpy
import sys
from time import sleep

from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool

from robot_sim.srv import LoadMap, ResetMap, SetRobotPose
from robot_sim.action import NavigateToPose

COMPLETION_PERCENTAGE = 0.999 # of known full map

MAP_PATH = ['./src/cas726_project/bitmap_maps/map-0.bmp',
            './src/cas726_project/bitmap_maps/map-1.bmp',
            './src/cas726_project/bitmap_maps/map-2.bmp']

MAP_NUM = 1

class Human_Explorer_Sim(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__('human_explorer_sim')

        # Set up publishers, subscribers and clients
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 1)
        self.pose_subscriber = self.create_subscription(PointStamped, '/clicked_point', self.clicked_point_callback, 1)
        self.evaluation_publisher = self.create_publisher(Bool, '/evaluation', 1)

        self.sim_nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.sim_load_map_client = self.create_client(LoadMap, 'load_map')
        self.sim_reset_map_client = self.create_client(ResetMap, 'reset_map')
        self.sim_set_pose_client = self.create_client(SetRobotPose, 'set_robot_pose')

        # Initialize environment variables
        self.map_msg = None

        # Wait for simulator to fully activate
        print('Waiting for simulation services')
        self.sim_nav_client.wait_for_server()
        self.sim_load_map_client.wait_for_service()
        self.sim_reset_map_client.wait_for_service()
        self.sim_set_pose_client.wait_for_service()

        # Load the map into the simulator
        self.load_map(MAP_NUM)

        print('Human Explorer Node initialized')
 
    
    def load_map(self, map_num):
        # Load the map into the custom simulator
        req = LoadMap.Request()
        req.map_path = MAP_PATH[map_num]
        req.threshold = 200
        req.resolution = 0.05
        req.flip = False
        req.rotate = 0
        future = self.sim_load_map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if (not res.success):
            print('Failed to load map')
            sys.exit(1)

        self.max_free_pixels = round(np.sum(np.array(res.map.data) == 0))

        # Set Robot Pose in the Simulator
        req = SetRobotPose.Request()
        req.pose = Pose()
        req.pose.position.x = 0.0
        req.pose.position.y = 0.0
        future = self.sim_set_pose_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if (not res.success):
            print('Failed to set pose')
            sys.exit(1)


    def go_to_pose(self, x, y):
        print(f'Sending goal pose x:{x:.2f}, y:{y:.2f}')
        # Send the goal pose to the simulator
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.speed = 0.6 # m/s
        self.sim_nav_client.send_goal_async(goal_msg)


    def check_done(self):
        free_pixels = np.sum((np.array(self.map_msg.data) == 0).astype(int))
        return bool(free_pixels > COMPLETION_PERCENTAGE*self.max_free_pixels)


    # Callbacks

    def map_callback(self, map_msg):
        self.map_msg = map_msg
    
    def goal_pose_callback(self, pose_msg):
        self.go_to_pose(pose_msg.pose.position.x, pose_msg.pose.position.y)

    def clicked_point_callback(self, point_msg):
        self.go_to_pose(point_msg.point.x, point_msg.point.y)


def main(args=None):
    rclpy.init(args=args)

    human_explorer = Human_Explorer_Sim()

    while (human_explorer.map_msg is None):
        print('Waiting for reset')
        rclpy.spin_once(human_explorer)

    # Tell the evaluator to start recording progress
    eval_status = Bool()
    eval_status.data = True
    human_explorer.evaluation_publisher.publish(eval_status)

    start_time = human_explorer.get_clock().now().nanoseconds

    while not human_explorer.check_done():
        rclpy.spin_once(human_explorer)

    stop_time = human_explorer.get_clock().now().nanoseconds

    # Grace period for simulator / evaluator to catch up
    sleep(5)

    # Tell the evaluator to stop recording progress
    eval_status.data = False
    human_explorer.evaluation_publisher.publish(eval_status)

    duration = (stop_time - start_time)/1e9
    print(f'Exploration took {duration} seconds')

    human_explorer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

