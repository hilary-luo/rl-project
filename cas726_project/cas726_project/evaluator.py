import numpy as np
import rclpy

from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid

class Evaluator(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__('evaluator')

        # Set up publishers, subscribers and clients
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)

        print('Evaluator Node initialized')

    # Callbacks

    def map_callback(self, map_msg):
        # TODO: Track progress (new area discovered over time - to be graphed)
        ''

def main(args=None):
    rclpy.init(args=args)
    evaluator = Evaluator()
    rclpy.spin(evaluator)
    evaluator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

