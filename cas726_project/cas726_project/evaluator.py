# Monitor and record the progression of map exploration
# ros2 run cas726_project evaluator <data_file_name>

import numpy as np
import rclpy

from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Bool
import csv
import sys

DATA_RECORD_INTERVAL = 1 # seconds
USE_CUSTOM_SIM = True # set to true to use with custom simulator, false if using with TurtleBot simulator

class Evaluator(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__('evaluator')

        # Set up publishers, subscribers and clients
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        if USE_CUSTOM_SIM:
            self.pose_subscriber = self.create_subscription(PoseStamped, '/pose', self.pose_callback, 1)
        else:
            self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.evaluation_subscriber = self.create_subscription(Bool, '/evaluation', self.eval_callback, 1)
        self.path_publisher = self.create_publisher(Path, '/robot_path', 10)
        self.timer_ = self.create_timer(DATA_RECORD_INTERVAL , self.timer_callback)
        
        # Initialize member variables
        self.path = Path()
        self.path.header.frame_id = '/map'
        self.starting_time = self.get_clock().now().nanoseconds
        self.prev_map = None
        self.current_map = None
        self.active = False

        # Create a file to record the progression data
        self.file_name = f'{str(sys.argv[1])}.csv' # Receive the file name as argument
        with open(self.file_name, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['Time', 'New Area', 'Total Area'])
            
        print('Evaluator Node initialized')


    def reset(self):
        self.path.poses.clear()

    # Evaluate how many new pixels have been mapped and save it in the csv
    def compute_new_area(self):
        prev_area = np.sum((np.array(self.prev_map.data) == 0).astype(int))
        current_area = np.sum((np.array(self.current_map.data) == 0).astype(int))
        new_area = current_area - prev_area 

        with open(self.file_name , mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([round((self.get_clock().now().nanoseconds - self.starting_time)/1e9, 2), new_area, current_area])

        return new_area
    
    # Callbacks

    # Receive start / end commands for path recording
    def eval_callback(self, eval_msg):
        self.active = bool(eval_msg.data)
        self.starting_time = self.get_clock().now().nanoseconds
        print('Evaluation status set to: ', self.active)
        print('Path length thus far: ', len(self.path.poses))

    def map_callback(self, map_msg):
        self.current_map = map_msg

    def pose_callback(self, pose_msg):
        if self.active:
            if USE_CUSTOM_SIM:
                self.path.poses.append(pose_msg)
            else:
                p = PoseStamped()
                p.header = pose_msg.header
                p.pose = pose_msg.pose.pose
                self.path.poses.append(p)
        self.path_publisher.publish(self.path)

    def timer_callback(self):
        if (not self.active) or (self.current_map is None):
            return
        if self.prev_map is not None:
            new_area = self.compute_new_area()
            self.get_logger().info(f"New area discovered: {new_area}")        
        self.prev_map = self.current_map

def main(args=None):
    rclpy.init(args=args)
    evaluator = Evaluator()
    rclpy.spin(evaluator)
    evaluator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
