# Frontier based exploration for use with custom simulator

import numpy as np
import rclpy
import sys

from copy import copy
from time import sleep

from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool

from skimage import measure

from robot_sim.srv import LoadMap, ResetMap, SetRobotPose, ComputePath
from robot_sim.action import NavigateToPose

MIN_REGION_AREA = 10
FRONTIER_TRIGGER_DISTANCE = 0.3
COMPLETION_PERCENTAGE = 0.999 # of known full map

MAP_PATH = ['./src/cas726_project/bitmap_maps/map-0.bmp',
            './src/cas726_project/bitmap_maps/map-1.bmp',
            './src/cas726_project/bitmap_maps/map-2.bmp']

MAP_NUM = 1

#Enums for Frontier List
ENUM_PATH_DISTANCE = 0
ENUM_AREA = 1
ENUM_X = 2
ENUM_Y = 3
ENUM_LABEL = 4
ENUM_PATH = 5

class Frontier_Explorer_Sim(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__('frontier_explorer_sim')

        # Set up publishers, subscribers and clients
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, '/costmap', self.costmap_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseStamped, '/pose', self.pose_callback, 1)
        self.publisher_free = self.create_publisher(OccupancyGrid, '/free_cell_mask', 10)
        self.publisher_unknown = self.create_publisher(OccupancyGrid, '/unknown_cell_mask', 10)
        self.publisher_frontier = self.create_publisher(OccupancyGrid, '/frontier_cells', 10)
        self.publisher_path = self.create_publisher(Path, '/selected_path', 10)
        self.evaluation_publisher = self.create_publisher(Bool, '/evaluation', 1)
        self.sim_nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.sim_load_map_client = self.create_client(LoadMap, 'load_map')
        self.sim_reset_map_client = self.create_client(ResetMap, 'reset_map')
        self.sim_set_pose_client = self.create_client(SetRobotPose, 'set_robot_pose')
        self.sim_compute_path_client = self.create_client(ComputePath, 'compute_path')

        # Initialize environment variables
        self.map_msg = None
        self.costmap_msg = None
        self.pose_msg = None
        self.frontier_list = list()

        # Wait for simulator to fully activate
        print('Waiting for simulation services')
        self.sim_nav_client.wait_for_server()
        self.sim_load_map_client.wait_for_service()
        self.sim_reset_map_client.wait_for_service()
        self.sim_set_pose_client.wait_for_service()
        self.sim_compute_path_client.wait_for_service()

        # Load the map into the simulator
        self.load_map(MAP_NUM)

        print('Frontier Explorer Node initialized')

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

        costmap = self.get_costmap()
        safe_pixels = (np.logical_and((costmap < 10), (costmap >= 0))).astype(int) # '1' are the free cells
        
        for region in measure.regionprops(labeled_frontier_map):
            if region.area >= MIN_REGION_AREA:

                x_centroid = round(region.centroid[1])
                y_centroid = round(region.centroid[0])

                # Find the closest free cell to calculate the path to
                if (safe_pixels[y_centroid, x_centroid] != 1):
                    free_cells = np.argwhere(safe_pixels)
                    distances = abs(free_cells[:, 0] - y_centroid) + abs(free_cells[:, 1] - x_centroid)
                    nearest_free = np.argmin(distances)
                    x_centroid = free_cells[nearest_free][1]
                    y_centroid = free_cells[nearest_free][0]

                # Find the centre of the frontier
                x = x_centroid*self.map_msg.info.resolution+map_origin_x
                y = y_centroid*self.map_msg.info.resolution+map_origin_y

                req = ComputePath.Request()
                req.pose.position.x = x
                req.pose.position.y = y
                future = self.sim_compute_path_client.call_async(req)

                # Wait for the path computation to complete before taking the next step
                rclpy.spin_until_future_complete(self, future)
                res = future.result()
                if (not res.success):
                    print('Failed to compute path')
                
                else:
                    self.frontier_list.append( # List of tuples of (Distance, size, centroid x, centroid y, label, path)
                        (len(res.path.poses), 
                        region.area, 
                        x,
                        y,
                        region.label,
                        res.path))
                
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
        goal_msg.pose.position.x = frontier[ENUM_X]
        goal_msg.pose.position.y = frontier[ENUM_Y]
        goal_msg.speed = 0.3 # m/s
        goal_future = self.sim_nav_client.send_goal_async(goal_msg)

        # Wait for the navigation goal to be accepted before taking the next step
        rclpy.spin_until_future_complete(self, goal_future)


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


    def check_done(self):
        free_pixels = np.sum((np.array(self.map_msg.data) == 0).astype(int))
        return bool(free_pixels > COMPLETION_PERCENTAGE*self.max_free_pixels)


    # Getters

    def get_map(self):
        return np.array(self.map_msg.data, dtype=np.int8).reshape(self.map_msg.info.height, self.map_msg.info.width)
    
    def get_costmap(self):
        return np.array(self.costmap_msg.data, dtype=np.int8).reshape(self.costmap_msg.info.height, self.costmap_msg.info.width)

    def get_map_origin(self):
        return (self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y)
    

    # Callbacks

    def map_callback(self, map_msg):
        self.map_msg = map_msg

    def costmap_callback(self, costmap_msg):
        self.costmap_msg = costmap_msg
    
    def pose_callback(self, pose_msg):
        self.pose_msg = pose_msg


def main(args=None):
    rclpy.init(args=args)

    frontier_explorer = Frontier_Explorer_Sim()

    while ((frontier_explorer.map_msg is None) or (frontier_explorer.costmap_msg is None) or (frontier_explorer.pose_msg is None)):
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
        while ((frontier_explorer.map_msg.info.width != frontier_explorer.costmap_msg.info.width) or 
               (frontier_explorer.map_msg.info.height != frontier_explorer.costmap_msg.info.height)):
            rclpy.spin_once(frontier_explorer)

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

