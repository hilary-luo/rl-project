import numpy as np
import rclpy


from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseWithCovarianceStamped
import csv
import sys

class Evaluator(Node):
    def __init__(self, filename):
        # Initialize ROS node
        super().__init__('evaluator')

        # Set up publishers, subscribers and clients
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.path_publisher = self.create_publisher(Path, '/robot_path', 10)

        map_update_interval_ = self.declare_parameter("map_update_interval", 1.0).value
        
        self.active = False
        self.session_cnt = 0

        self.path = Path()
        self.path.header.frame_id = '/map'

        timer_ = self.create_timer(map_update_interval_ , self.timer_callback)
        self.starting_time = float(self.get_clock().now().to_msg().sec)
        self.prev_map = None
        self.current_map = None
        self.file_name = filename
        with open(f'{self.file_name}_{self.session_cnt}', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['Time', 'New Area'])
            


        print('Evaluator Node initialized')

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def reset(self):
        self.path = Path()
        self.session_cnt = self.session_cnt + 1
        with open(f'{self.file_name}_{self.session_cnt}', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['Time', 'New Area'])

    # Callbacks

    def map_callback(self, map_msg):
        self.current_map = map_msg

    def pose_callback(self, pose_msg):
        if self.active:
            self.path.poses.append(pose_msg.pose.pose)
        self.path_publisher.publish(self.path)

    def timer_callback(self):
        if not self.active:
            return
        if self.prev_map is None:
            self.prev_map = self.current_map
        else:
            current_map = self.current_map
            new_area = self.compute_new_area(current_map)
            print(new_area)

            self.get_logger().info(f"New area discovered: {new_area}")
        
            self.prev_map = current_map

    
    def compute_new_area(self, current_map):
        prev_map_data = self.prev_map.data
        current_map_data = current_map.data
        resolution = current_map.info.resolution
        width = current_map.info.width

        prev_free_cells = [i for i in range(len(prev_map_data)) if prev_map_data[i] == 0]
        current_free_cells = [i for i in range(len(current_map_data)) if current_map_data[i] == 0]

        prev_area = len(prev_free_cells) 
        current_area = len(current_free_cells)

        new_area = current_area - prev_area 

        with open(f'{self.file_name}_{self.session_cnt}', mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([float(self.get_clock().now().to_msg().sec)-self.starting_time, new_area, current_area])

        return new_area


def main(args=None):
    rclpy.init(args=args)
    evaluator = Evaluator()
    rclpy.spin(evaluator)
    evaluator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
