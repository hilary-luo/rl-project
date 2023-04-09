import numpy as np
import rclpy
import math

from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose

from skimage import measure
from skimage.measure import label, regionprops

MIN_REGION_AREA = 20

class Frontier_Explorer(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__('frontier_explorer')

        # Set up publishers, subscribers and clients
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.publisher_free = self.create_publisher(OccupancyGrid, '/free_cell_mask', 10)
        self.publisher_unknown = self.create_publisher(OccupancyGrid, '/unknown_cell_mask', 10)
        self.publisher_frontier = self.create_publisher(OccupancyGrid, '/frontier_cells', 10)

        # Initialize environment variables
        self.map_msg = None
        self.pose_msg = None
        self.costmap = None
        self.frontier_list = list()
        self.print = False

        self.start_time = self.get_clock().now()

        # Wait for navigation to fully activate
        print('Waiting for Nav2 server')
        self.nav2_client.wait_for_server()

        print('Frontier Explorer Node initialized')

    def find_frontiers(self):
        map = self.get_map()

        # Detect Frontiers in Free Cells
        free_map = (np.logical_and((map < 10), (map >= 0))).astype(int) # '1' are the open cells, '0' are the unknown or occupied cells
        free_frontiers = measure.find_contours(free_map, 0.5)
        free_cells = list()
        for f in free_frontiers:
            for c in f:
                free_cells.append(c.tolist())

        free_map_msg = OccupancyGrid()
        free_map_msg.header = self.map_msg.header
        free_map_msg.info = self.map_msg.info
        free_map_msg.data = (free_map.flatten()*100).tolist()
        self.publisher_free.publish(free_map_msg)

        # Detect Frontiers in Unknown Cells
        unknown_map = (map >= 0).astype(int) # 1 are the free or occupied cells, '0' are the unknown cells
        unknown_frontiers = measure.find_contours(unknown_map, 0.5)
        unknown_cells = list()
        for f in unknown_frontiers:
            for c in f:
                unknown_cells.append(c.tolist())

        unknown_map_msg = OccupancyGrid()
        unknown_map_msg.header = self.map_msg.header
        unknown_map_msg.info = self.map_msg.info
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
        for region in regionprops(labeled_frontier_map):
            if region.area >= MIN_REGION_AREA:
                # Find the centre of the frontier
                x = region.centroid[1]*self.map_msg.info.resolution+map_origin_x
                y = region.centroid[0]*self.map_msg.info.resolution+map_origin_y
                pose_x, pose_y = self.get_pose()
                self.frontier_list.append( # List of tuples of (Distance, size, centroid x, centroid y, label)
                    (math.sqrt((x - pose_x)**2 + (y - pose_y)**2), 
                     region.area, 
                     x,
                     y,
                     region.label)) 
                
        self.publish_frontiers(labeled_frontier_map)
        print('Frontier count: ',len(self.frontier_list))

    # Getters

    def get_map(self):
        return np.array(self.map_msg.data, dtype=np.int8).reshape(self.map_msg.info.height, self.map_msg.info.width)

    def get_map_origin(self):
        return (self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y)

    def get_pose(self):
        return (self.pose_msg.pose.pose.position.x, self.pose_msg.pose.pose.position.y)

    def publish_frontiers(self, labeled_frontier_map):
        labeled_map_np = np.array(labeled_frontier_map)
        processed_map = np.zeros(labeled_map_np.shape)
        n = 0
        local_list = self.frontier_list.copy()
        local_list.sort(reverse=True) # Order so pink will be the target
        for frontier in local_list:
            print(frontier)
            print(f'Distance is {frontier[0]}, Label is {frontier[4]}, n is {n}')
            n = n + 1
            processed_map[labeled_map_np == frontier[4]] = n

        frontier_map_msg = OccupancyGrid()
        frontier_map_msg.header = self.map_msg.header
        frontier_map_msg.info = self.map_msg.info
        frontier_map_msg.data = ((processed_map*100/n).flatten()).astype(int).tolist()
        self.publisher_frontier.publish(frontier_map_msg)


    def select_frontier(self):
        # TODO: Choose the frontier in a more effective way (currently going to closest)
        self.frontier_list.sort()
        selected_frontier = self.frontier_list[0]
        return selected_frontier

    def go_to_frontier(self, frontier):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = frontier[2]
        goal_pose.pose.position.y = frontier[3]
        goal_pose.pose.orientation.z = 0.0

        print(f"Selected Frontier Pose is {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} at distance {frontier[0]} and of size {frontier[1]}")

        # Send the goal pose to the navigation stack
        print('Sending the pose')
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
    
    def check_done(self):
        perc_pixels_visited = np.sum((np.array(self.map_msg.data) >= 0).astype(int)) / len(self.map_msg.data)
        print(f'Visited {perc_pixels_visited*100:.1f}% of pixels\n')
        return (bool(perc_pixels_visited > 0.60), self.start_time.nanoseconds, self.get_clock().now().nanoseconds)

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

    done = False

    while not done:
        frontier_explorer.find_frontiers()
        frontier = frontier_explorer.select_frontier()
        if(frontier is not None):
            #print(frontier_explorer.frontier_list)
            frontier_explorer.go_to_frontier(frontier)
            done, start_time, curr_time = frontier_explorer.check_done()
        rclpy.spin_once(frontier_explorer) # Allows callbacks to update the map
        
    duration = curr_time - start_time

    print(f'Exploration took {duration/1e9} seconds')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    frontier_explorer.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()

