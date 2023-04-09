import rclpy
import numpy as np

import gym
from gym import spaces
from rclpy.action import ActionClient

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose

import skimage.measure

MAP_MAX_POOLING = 2
AI_MAP_DIM = 256
LOG_DIR = './td3_project_monitor'
LOG_DIR_TENSORBOARD = './td3_project_tensorboard'
MODEL_CHECKPOINT_SAVE_PATH = './td3_nav_checkpoint'
MODEL_SAVE_PATH = 'td3_nav_model'
MODEL_LOAD_PATH = 'td3_nav_model'

# Custom Gym environment for navigation task
class NavEnv(gym.Env, Node):

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # Initialize Node
        super().__init__('nav_env')

        # Define the observation and action space for the environment
        self.observation_space = spaces.Dict({
            'map': spaces.Box(low=0, high=255, shape=(AI_MAP_DIM, AI_MAP_DIM, 1), dtype=np.uint8), # 255 represents unknown space in order to satisfy uint8
            'robot_pose': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        })

        # AI output will be an xy position on the map with a resolution of every 4 pixels
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Initialize environment variables
        self.map = None
        self.costmap = None
        self.visited_pixels = None
        self.robot_pose = None
        self.map_reset_status = True
        self.pose_reset_status = True
        self.costmap_reset_status = True

        # Set up publishers, subscribers and clients
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.pose_subscriber = self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 1)

        # Wait for navigation to fully activate
        print('Waiting for Nav2 server')
        self.nav2_client.wait_for_server()

        print('Environment Node initialized')

    def isReady(self):
        return (not self.map_reset_status and not self.pose_reset_status and not self.costmap_reset_status)

    def reset(self): #TODO: Reset the Nav2 map
        # Reset the environment to its initial state and return an observation
        self.map = None
        self.costmap = None
        self.visited_pixels = np.zeros((AI_MAP_DIM, AI_MAP_DIM), dtype=np.int8)
        self.robot_pose = None
        self.map_reset_status = True
        self.pose_reset_status = True
        self.costmap_reset_status = True
        while rclpy.ok() and not self.isReady():
            print('waiting for reset')
            rclpy.spin_once(self)
        print('Environment reset')
        return self._get_obs()

    def step(self, action):
        # Save pose before moving
        robot_pose_prev = self.robot_pose
        robot_pose_in_action = self.map_frame_to_action_space(robot_pose_prev)
        action_x_prev = round((robot_pose_in_action[0].item()+1)*(AI_MAP_DIM-1)/2.0)
        action_y_prev = round((robot_pose_in_action[1].item()+1)*(AI_MAP_DIM-1)/2.0)

        # Flag to represent whether the robot can reach the goal given by the model
        invalid_goal = False

        # Set the next goal pose for Nav2
        goal_pose = self.action_space_to_map_frame(action)
        map_2d_array = self.map_msg_to_obs_space(self.map, squeeze=True)
        costmap_2d_array = self.map_msg_to_obs_space(self.costmap, squeeze=True)
        np.savetxt("/home/hluo/Downloads/map_obs_space.csv", map_2d_array, delimiter = ",")
        np.savetxt("/home/hluo/Downloads/costmap_obs_space.csv", costmap_2d_array, delimiter = ",")
        np.savetxt("/home/hluo/Downloads/map_full.csv",np.array(self.map.data, dtype=np.int8).reshape(self.map.info.height, self.map.info.width), delimiter = ",")
        np.savetxt("/home/hluo/Downloads/costmap_full.csv",np.array(self.costmap.data, dtype=np.int8).reshape(self.costmap.info.height, self.costmap.info.width), delimiter = ",")

        action_x = round((action[0].item()+1)*(AI_MAP_DIM-1)/2.0)
        action_y = round((action[1].item()+1)*(AI_MAP_DIM-1)/2.0)

        # Determine if the robot should move to the goal
        print(f"Goal Pose is {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} aka action {action_x} {action_y}")

        # TODO: Do I need to check for a particular radius of not unknown
        if (map_2d_array[action_y][action_x] == -1):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is unknown space, skipping navigation')
            invalid_goal = True
            
        elif (map_2d_array[action_y][action_x] >= 50):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is occupied space, skipping navigation')
            invalid_goal = True

        # TODO: what should the threshold be for the costmap to avoid the blue area
        elif (costmap_2d_array[action_y][action_x] >= 20):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is near occupied space, skipping navigation')
            invalid_goal = True
            print(f"Occupancy value: {map_2d_array[action_y][action_x]}    Costmap value {costmap_2d_array[action_y][action_x]}")

        # TODO: what should the threshold be for the costmap to avoid the blue area
        elif (abs(action_x - action_x_prev) < 5 and abs(action_y - action_y_prev) < 5):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is too close to the current position, skipping navigation')
            invalid_goal = True

        else:
            print(f"Action x: {action_x}  Old position x: {action_x_prev} Action y: {action_y}  Old position y: {action_y_prev}")
            print(f"Occupancy value: {map_2d_array[action_y][action_x]}    Costmap value {costmap_2d_array[action_y][action_x]}")
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

        done = False

        # Calculate reward
        if invalid_goal:
            reward = -1
        else:
            # Compute the reward (number of new pixels mapped - penalty for travel distance)
            old_pixels = self.visited_pixels.copy() #TODO: Update visited pixels on the first map / should sit for a couple of seconds and then set this and then start -> maybe handle this in main
            self.visited_pixels = (map_2d_array >= 0).astype(int)
            new_pixels = self.visited_pixels - old_pixels
            reward = min(1,np.sum(new_pixels)/5000) - min(1,(np.linalg.norm(np.array([robot_pose_in_action[0], robot_pose_in_action[1]]) - np.array([action[0].item(), action[1].item()]))/4))
            # TODO: fix reward multipliers
            
            # Check if the goal is reached
            perc_pixels_visited = (np.sum(self.visited_pixels))/(AI_MAP_DIM*AI_MAP_DIM)
            done = bool(perc_pixels_visited > 0.40)
            print(f"Reward Components - reward: {np.sum(new_pixels)/5000} Penalty: {np.linalg.norm(np.array([robot_pose_in_action[0], robot_pose_in_action[1]]) - np.array([action[0].item(), action[1].item()]))/4}")
            print(f'Reward: {reward:.3f}              Done: {done} - visited {np.sum(new_pixels)} new pixels making a total of {perc_pixels_visited*100:.1f}% of pixels\n')
            np.savetxt("/home/hluo/Downloads/visited_pixels.csv",np.array(self.visited_pixels, dtype=np.int8).reshape(AI_MAP_DIM, AI_MAP_DIM), delimiter = ",")

        # Update the observation (occupancy grid map and robot pose)
        obs = self._get_obs()
        #print(f'Observation: {obs}')

        return obs, reward, done, {'info':42}

    def render(self, mode="human"):
        print('render function')

    def close(self):
        print('close')

    def _get_obs(self):
        return {'map': self.map_msg_to_obs_space(self.map), 'robot_pose': self.map_frame_to_action_space(self.robot_pose)}

    def map_msg_to_obs_space(self, map_msg, squeeze=False):
        if (map_msg is None):
            return None
        # print(f'Map is {map_msg.info.height} by {map_msg.info.width}')
        # Convert the occupancy grid map message to a numpy array
        map_2d_array = np.array(map_msg.data, dtype=np.int8).reshape(map_msg.info.height, map_msg.info.width)
        padded_map = np.full((AI_MAP_DIM*MAP_MAX_POOLING, AI_MAP_DIM*MAP_MAX_POOLING), -1, dtype=np.int8)
        #TODO: Insert check to make sure it isn't too big
        padded_map[0:map_2d_array.shape[0],0:map_2d_array.shape[1]] = map_2d_array
        
        # Resample the map with max pooling and then pad it with -1's to the full size
        obs_space_map = skimage.measure.block_reduce(padded_map,(MAP_MAX_POOLING,MAP_MAX_POOLING), np.max)
        if (squeeze):
            return obs_space_map
        
        return np.expand_dims(obs_space_map.astype(np.uint8),axis=2)


    def map_frame_to_action_space(self, pose_in):
        # Going from actual map frame (m, m, rad) to the AI Obs / Action format (0-64, 0-64, rad)
        x = (2*(pose_in.pose.pose.position.x - self.map.info.origin.position.x)/(self.map.info.resolution*MAP_MAX_POOLING*(AI_MAP_DIM-1))) - 1
        y = (2*(pose_in.pose.pose.position.y - self.map.info.origin.position.y)/(self.map.info.resolution*MAP_MAX_POOLING*(AI_MAP_DIM-1))) - 1
        # z = (pose_in.pose.orientation.z - self.map.info.origin.orientation.z)
        return np.array([x, y], dtype=np.float32)
    
    def action_space_to_map_frame(self, action):
        pose_out = PoseStamped()
        pose_out.header.frame_id = 'map'
        pose_out.pose.position.x = (round((action[0].item()+1)*(AI_MAP_DIM-1)/2.0)*self.map.info.resolution*MAP_MAX_POOLING) + self.map.info.origin.position.x
        pose_out.pose.position.y = (round((action[1].item()+1)*(AI_MAP_DIM-1)/2.0)*self.map.info.resolution*MAP_MAX_POOLING) + self.map.info.origin.position.y
        pose_out.pose.orientation.z = 0.0 + self.map.info.origin.orientation.z
        return pose_out

    # Callbacks

    def map_callback(self, map_msg):
        # print(f'Map is {map_msg.info.height} by {map_msg.info.width}')
        self.map = map_msg
        self.map_reset_status = False
        #print('Map Callback')
        if (self.visited_pixels is None):
            self.visited_pixels = self.map_msg_to_obs_space(self.map, squeeze=True)

    def costmap_callback(self, costmap_msg):
        self.costmap = costmap_msg
        self.costmap_reset_status = False
        #print('Costmap Callback')

    def pose_callback(self, pose_msg):
        self.robot_pose = pose_msg
        self.pose_reset_status = False
        #print('Pose Callback')


    def nav_callback(self, msg):
        #print('Nav Callback')
        ''

    def __del__(self):
        # Cleanup the ROS node
        self.destroy_node()