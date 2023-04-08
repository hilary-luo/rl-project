import numpy as np
import rclpy

from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose

from skimage import measure
from skimage.measure import label, regionprops

class Frontier_Explorer(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__('frontier_explorer')


        # Set up publishers, subscribers and clients
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.pose_subscriber = self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 1)

        # Initialize environment variables
        self.map_msg = None
        self.pose_msg = None
        self.costmap = None
        self.frontier_list = list()

        # Wait for navigation to fully activate
        print('Waiting for Nav2 server')
        self.nav2_client.wait_for_server()

        print('Frontier Explorer Node initialized')

    def find_frontiers(self):
        map = self.get_map()

        # Detect Frontiers in Open Cells
        open_map = ((map < 10) and (map >= 0)).astype(int) # '1' are the open cells, '0' are the unknown or occupied cells
        open_frontiers = measure.find_contours(open_map, 0.5)
        open_cells = list()
        for f in open_frontiers:
            for c in f:
                open_cells.append(c.tolist())

        # Detect Frontiers in Unknown Cells
        unknown_map = (map > 0).astype(int) # 1 are the free or occupied cells, '0' are the unknown cells
        unknown_frontiers = measure.find_contours(unknown_map, 0.5)
        unknown_cells = list()
        for f in unknown_frontiers:
            for c in f:
                unknown_cells.append(c.tolist())

        # Cross check between the frontiers to find the frontiers for exploration
        frontier_cells = [c for c in unknown_cells if c in open_cells]

        frontier_map = map * 0
        for cell in frontier_cells:
            frontier_map[int(cell[0]), int(cell[1])]

        # Cluster cells into frontiers
        labeled_frontier_map, label_num = measure.label(frontier_map, return_num=True)
        map_origin_x, map_origin_y = self.get_map_origin()

        for region in regionprops(labeled_frontier_map):
            if region.area >= 20:
                # Find the centre of the frontier
                x = region.centroid[1]
                y = region.centroid[0]
                pose_x, pose_y = self.get_pose()
                self.frontier_list.append( # Tuple of Manhatten distance, x, y, size
                    (abs(x - pose_x) + abs(y - pose_y), 
                     region.area, 
                     x*self.map.info.resolution+map_origin_x,
                     y*self.map.info.resolution+map_origin_y)) 
                self.frontier_list.sort(reverse=True)

    def get_map(self):
        return np.array(self.map_msg.data, dtype=np.int8).reshape(self.map_msg.info.height, self.map_msg.info.width)

    def get_map_origin(self):
        return (self.map.info.origin.position.x, self.map.info.origin.position.y)

    def get_pose(self):
        return (self.pose_msg.pose.pose.position.x, self.pose_msg.pose.pose.position.y)

    def select_frontier(self):
        # 
        ''
    
    def go_to_frontier(self):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = self.frontier_list[0][2]
        goal_pose.pose.position.y = self.frontier_list[0][3]
        goal_pose.pose.orientation.z = 0.0

        # Send the goal pose to the navigation stack
        #print('Sending the pose')
        nav2_goal_msg = NavigateToPose.Goal()
        nav2_goal_msg.pose = goal_pose
        nav2_future = self.nav2_client.send_goal_async(nav2_goal_msg, self.nav_callback)

        # Wait for the navigation to complete before taking the next step
        #print('Checking goal acceptance')
        rclpy.spin_until_future_complete(self, nav2_future)
        goal_handle = nav2_future.result()

        while (not goal_handle.accepted):
            print('Goal was rejected')
            nav2_future = self.nav2_client.send_goal_async(nav2_goal_msg, self.nav_callback)
            rclpy.spin_until_future_complete(self, nav2_future)
            goal_handle = nav2_future.result()

        result_future = goal_handle.get_result_async()

        print('Waiting for navigation to complete')
        rclpy.spin_until_future_complete(self,result_future)
        print('Navigation Completed')
    
    # Callbacks

    def map_callback(self, map_msg):
        # print(f'Map is {map_msg.info.height} by {map_msg.info.width}')
        self.map_msg = map_msg
        self.map_reset_status = False
        #print('Map Callback')

    def costmap_callback(self, costmap_msg):
        self.costmap = costmap_msg
        self.costmap_reset_status = False
        #print('Costmap Callback')

    
    def pose_callback(self, pose_msg_in):
        self.pose_msg = pose_msg_in
        self.pose_reset_status = False
        #print('Pose Callback')


    def nav_callback(self, msg):
        #print('Nav Callback')
        ''

def main(args=None):
    rclpy.init(args=args)

    frontier_explorer = Frontier_Explorer()

    while ((frontier_explorer.map_msg is None) or (frontier_explorer.pose_msg is None)):
        print('waiting for reset')
        rclpy.spin_once(frontier_explorer)

    frontier_explorer.find_frontiers()
    frontier_explorer.select_frontier()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    frontier_explorer.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()

