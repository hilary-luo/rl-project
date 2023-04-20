# Frontier based exploration for use with Turtlebot 4 simulator
# ros2 run cas726_project frontier_explorer

import numpy as np
import rclpy
from copy import copy
from time import sleep
from PIL import Image

from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose, ComputePathThroughPoses
from std_msgs.msg import Bool

from skimage import measure

MIN_REGION_AREA = 10
FRONTIER_TRIGGER_DISTANCE = 0.3
COMPLETION_PERCENTAGE = 0.95 # of known full map

#Enums for Frontier List
ENUM_PATH_DISTANCE = 0
ENUM_AREA = 1
ENUM_X = 2
ENUM_Y = 3
ENUM_LABEL = 4
ENUM_PATH = 5

class Frontier_Explorer(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__('frontier_explorer')

        # Set up publishers, subscribers and clients
        self.nav2_nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.nav2_path_client = ActionClient(self, ComputePathThroughPoses, 'compute_path_through_poses')
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.publisher_free = self.create_publisher(OccupancyGrid, '/free_cell_mask', 10)
        self.publisher_unknown = self.create_publisher(OccupancyGrid, '/unknown_cell_mask', 10)
        self.publisher_frontier = self.create_publisher(OccupancyGrid, '/frontier_cells', 10)
        self.publisher_path = self.create_publisher(Path, '/selected_path', 10)
        self.evaluation_publisher = self.create_publisher(Bool, '/evaluation', 1)

        # Initialize environment variables
        self.map_msg = None
        self.pose_msg = None
        self.frontier_list = list()
        self.max_pixels = self.determineDoneCondition("./maps/maze-ttb-sim.pgm")

        # Wait for navigation to fully activate
        print('Waiting for Nav2 server')
        self.nav2_nav_client.wait_for_server()

        print('Frontier Explorer Node initialized')

    def determineDoneCondition(self, path):
        full_map = np.asarray(Image.open(path))
        max_pixels = np.sum(np.logical_or(full_map < 180,full_map > 220))
        print(f'Max number of pixels to map is {max_pixels} or {max_pixels*100/(full_map.shape[0]*full_map.shape[1]):.2f}% of total pixels')
        return max_pixels
    
    def find_frontiers(self):
        map = self.get_map()

        # Detect Frontiers in Free Cells
        free_map = (np.logical_and((map < 10), (map >= 0))).astype(int) # '1' are the free cells, '0' are the unknown or occupied cells
        free_frontiers = measure.find_contours(free_map, 0.5)
        free_cells = list()
        for f in free_frontiers:
            for c in f:
                free_cells.append(c.tolist())

        free_map_msg = OccupancyGrid()
        free_map_msg.header = copy(self.map_msg.header)
        free_map_msg.info = copy(self.map_msg.info)
        free_map_msg.data = (free_map.flatten()*100).tolist()
        self.publisher_free.publish(free_map_msg)

        # Detect Frontiers in Unknown Cells
        unknown_map = (map >= 0).astype(int) # '1' are the free or occupied cells, '0' are the unknown cells
        unknown_frontiers = measure.find_contours(unknown_map, 0.5)
        unknown_cells = list()
        for f in unknown_frontiers:
            for c in f:
                unknown_cells.append(c.tolist())

        unknown_map_msg = OccupancyGrid()
        unknown_map_msg.header = copy(self.map_msg.header)
        unknown_map_msg.info = copy(self.map_msg.info)
        unknown_map_msg.data = (unknown_map.flatten()*100).tolist()
        self.publisher_unknown.publish(unknown_map_msg)

        # Cross check between the frontiers to find the frontiers for exploration (ignore frontiers between free and occupied and unknown and occupied)
        frontier_cells = [c for c in unknown_cells if c in free_cells]

        frontier_map = np.zeros(map.shape)
        for cell in frontier_cells:
            frontier_map[int(cell[0]), int(cell[1])] = 1

        # Cluster cells into frontiers
        labeled_frontier_map = measure.label(frontier_map)
        map_origin_x, map_origin_y = self.get_map_origin()

        # Populate the list of all frontiers to be considered and the required metrics
        self.frontier_list.clear()
        for region in measure.regionprops(labeled_frontier_map):
            if region.area >= MIN_REGION_AREA:
                # Find the centre of the frontier
                x = region.centroid[1]*self.map_msg.info.resolution+map_origin_x
                y = region.centroid[0]*self.map_msg.info.resolution+map_origin_y
                pose_temp = PoseStamped()
                pose_temp.header = copy(self.pose_msg.header)
                pose_temp.pose.position.x = x
                pose_temp.pose.position.y = y
                nav2_goal_msg = ComputePathThroughPoses.Goal()
                nav2_goal_msg.goals = [pose_temp]
                nav2_goal_msg.use_start = False
                nav2_future = self.nav2_path_client.send_goal_async(nav2_goal_msg)

                # Wait for the path computation to complete before taking the next step
                rclpy.spin_until_future_complete(self, nav2_future)
                goal_handle = nav2_future.result()

                while (not goal_handle.accepted):
                    print('Path was rejected')
                    nav2_future = self.nav2_path_client.send_goal_async(nav2_goal_msg)
                    rclpy.spin_until_future_complete(self, nav2_future)
                    goal_handle = nav2_future.result()

                result_future = goal_handle.get_result_async()

                #print('Waiting for path planning to complete')
                rclpy.spin_until_future_complete(self,result_future)

                if (len(result_future.result().result.path.poses) > 0):
                    self.frontier_list.append( # List of tuples of (Distance, size, centroid x, centroid y, label, path)
                        (len(result_future.result().result.path.poses), 
                        region.area, 
                        x,
                        y,
                        region.label,
                        result_future.result().result.path)) 
                    
                
        self.publish_frontiers(labeled_frontier_map, map)
        print('Frontier count: ',len(self.frontier_list))


    def publish_frontiers(self, labeled_frontier_map, map):
        labeled_map_np = np.array(labeled_frontier_map)
        processed_map = np.zeros(labeled_map_np.shape)
        n = 0
        local_list = self.frontier_list.copy()
        local_list.sort(reverse=True) # Order so pink will be the target
        for frontier in local_list:
            n = n + 1
            processed_map[labeled_map_np == frontier[ENUM_LABEL]] = n

        if (n>0):
            frontier_map_msg = OccupancyGrid()
            frontier_map_msg.header = copy(self.map_msg.header)
            frontier_map_msg.info = copy(self.map_msg.info)
            frontier_map_msg.info.height = map.shape[0]
            frontier_map_msg.info.width = map.shape[1]
            frontier_map_msg.data = ((processed_map*100/n).flatten()).astype(int).tolist()
            self.publisher_frontier.publish(frontier_map_msg)


    def select_frontier(self):
        if (len(self.frontier_list) == 0):
            return None
        self.frontier_list.sort()
        selected_frontier = self.frontier_list[0]
        return selected_frontier


    def go_to_frontier(self, frontier):
        print(f"Selected Frontier Pose is {frontier[ENUM_X]:.2f}, {frontier[ENUM_Y]:.2f} at distance {frontier[ENUM_PATH_DISTANCE]} and of size {frontier[ENUM_AREA]}")

        # Send the goal pose to the simulator
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = frontier[ENUM_X]
        goal_msg.pose.pose.position.y = frontier[ENUM_Y]
        goal_future = self.nav2_nav_client.send_goal_async(goal_msg)

        # Wait for the navigation goal to be accepted before taking the next step
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()

        while (not goal_handle.accepted):
            print('Goal was rejected')
            goal_future = self.nav2_nav_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, goal_future)
            goal_handle = goal_future.result()


    def check_done(self):
        pixels_visited = np.sum((np.array(self.map_msg.data) >= 0).astype(int))
        return bool(pixels_visited > COMPLETION_PERCENTAGE*self.max_pixels)


    # Getters

    def get_map(self):
        return np.array(self.map_msg.data, dtype=np.int8).reshape(self.map_msg.info.height, self.map_msg.info.width)

    def get_map_origin(self):
        return (self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y)
    

    # Callbacks

    def map_callback(self, map_msg):
        self.map_msg = map_msg
    
    def pose_callback(self, pose_msg):
        self.pose_msg = pose_msg


def main(args=None):
    rclpy.init(args=args)

    frontier_explorer = Frontier_Explorer()

    while ((frontier_explorer.map_msg is None) or (frontier_explorer.pose_msg is None)):
        print('Waiting for reset')
        rclpy.spin_once(frontier_explorer)

    old_frontier = None

    # Tell the evaluator to start recording progress
    eval_status = Bool()
    eval_status.data = True
    frontier_explorer.evaluation_publisher.publish(eval_status)

    start_time = frontier_explorer.get_clock().now().nanoseconds

    while not frontier_explorer.check_done():
        frontier_explorer.find_frontiers()
        frontier = frontier_explorer.select_frontier()
        if(frontier is not None):
            print(f'New frontier is {frontier[ENUM_X]}, {frontier[ENUM_Y]}')
        else:
            print('New frontier is none')
        if((frontier is not None) and ((old_frontier is None) or (abs(frontier[ENUM_X]-old_frontier[ENUM_X])+abs(frontier[ENUM_Y]-old_frontier[ENUM_Y]))>FRONTIER_TRIGGER_DISTANCE)):
            print(f'New frontier to explore at {frontier[ENUM_X]}, {frontier[ENUM_Y]}')
            frontier_explorer.go_to_frontier(frontier)
            old_frontier = frontier
            frontier_explorer.publisher_path.publish(frontier[ENUM_PATH])

        rclpy.spin_once(frontier_explorer) # Allows callbacks to update the map

    stop_time = frontier_explorer.get_clock().now().nanoseconds

    # Grace period for simulator / evaluator to catch up
    sleep(5)

    # Tell the evaluator to stop recording progress
    eval_status.data = False
    frontier_explorer.evaluation_publisher.publish(eval_status)

    duration = (stop_time - start_time)/1e9
    print(f'Exploration took {duration} seconds')

    frontier_explorer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

