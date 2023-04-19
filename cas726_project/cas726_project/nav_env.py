# Reinforcement learning environment for autonomous exploration for use with Turtlebot 4 simulator

import rclpy
import numpy as np
from PIL import Image

import gym
from gym import spaces
from rclpy.action import ActionClient
import csv

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool

import skimage.measure

MAP_MAX_POOLING = 2 # Scaling factor to reduce map size for the observation space
AI_MAP_DIM = 256 # Environment can only support maps of max size AI_MAP_DIM*MAP_MAX_POOLING

COMPLETION_PERCENTAGE = 0.9 # of known full map
LEARNING_RECORD_PATH = f'./action_record/ai_action_record_{round(COMPLETION_PERCENTAGE*100)}'

# Custom Gym environment for navigation task
class NavEnv(gym.Env, Node):

    # For gym environment
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # Initialize Node
        super().__init__('nav_env')

        # Define the observation and action space for the environment
        self.observation_space = spaces.Dict({
            'map': spaces.Box(low=0, high=255, shape=(AI_MAP_DIM, AI_MAP_DIM, 1), dtype=np.uint8), # 255 represents unknown space in order to satisfy uint8
            'robot_pose': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        })

        # AI output will be an xy position normalized from -1 to 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Initialize environment variables
        self.map = None
        self.costmap = None
        self.visited_pixels = None
        self.robot_pose = None
        self.map_reset_status = True
        self.pose_reset_status = True
        self.costmap_reset_status = True
        self.max_pixels = self.determineDoneCondition("./maps/maze.pgm")

        # Set up publishers, subscribers and clients
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 1)
        self.evaluation_publisher = self.create_publisher(Bool, '/evaluation', 1)

        # Wait for navigation to fully activate
        print('Waiting for Nav2 server')
        self.nav2_client.wait_for_server()

        # Create csv file to record actions
        with open(LEARNING_RECORD_PATH, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['Action X', 'Action Y', 'Reward'])

        print('Environment Node initialized')

    def determineDoneCondition(self, path):
        full_map = np.asarray(Image.open(path))
        max_pixels = np.sum(np.logical_or(full_map < 180,full_map > 220))
        print(f'Max number of pixels to map is {max_pixels} or {max_pixels*100/(full_map.shape[0]*full_map.shape[1]):.2f}% of total pixels')
        return max_pixels

    def isReady(self):
        # Verify that all of the messages have been received since reset
        return (not self.map_reset_status and not self.pose_reset_status and not self.costmap_reset_status)

    def reset(self): #Does not reset the SLAM map -> must be done manually
        # Reset the environment to its initial state and return an observation
        self.map = None
        self.costmap = None
        self.visited_pixels = np.zeros((AI_MAP_DIM, AI_MAP_DIM), dtype=np.int8)
        self.robot_pose = None
        self.map_reset_status = True
        self.pose_reset_status = True
        self.costmap_reset_status = True

        print('Waiting for reset')
        while rclpy.ok() and not self.isReady():
            rclpy.spin_once(self)

        print('Environment reset')
        return self._get_obs()

    def step(self, action):
        rclpy.spin_once(self) # catch up on callbacks
        # Save pose before moving
        robot_pose_in_action = self.map_frame_to_action_space(self.robot_pose)
        action_x_prev = round((robot_pose_in_action[0].item()+1)*(AI_MAP_DIM-1)/2.0)
        action_y_prev = round((robot_pose_in_action[1].item()+1)*(AI_MAP_DIM-1)/2.0)

        # Flag to represent whether the robot can reach the goal given by the model
        invalid_goal = False
        
        # Set the next goal pose for Nav2
        goal_pose = self.action_space_to_map_frame(action)
        map_2d_array = self.map_msg_to_obs_space(self.map, squeeze=True)
        costmap_2d_array = self.map_msg_to_obs_space(self.costmap, squeeze=True)
        
        # Translate the action into an xy map position
        action_position_x = round((action[0].item()+1)*(AI_MAP_DIM-1)/2.0)
        action_position_y = round((action[1].item()+1)*(AI_MAP_DIM-1)/2.0)
        #print(f"Goal Pose is {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} aka action {action_position_x} {action_position_y}")
        # Determine if the robot should move to the goal
        if (map_2d_array[action_position_y][action_position_x] == -1):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is unknown space, skipping navigation')
            invalid_goal = True
            
        elif (map_2d_array[action_position_y][action_position_x] >= 50):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is occupied space, skipping navigation')
            invalid_goal = True

        elif (costmap_2d_array[action_position_y][action_position_x] >= 20):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is near occupied space, skipping navigation')
            invalid_goal = True
            #print(f"Occupancy value: {map_2d_array[action_position_y][action_position_x]}    Costmap value {costmap_2d_array[action_position_y][action_position_x]}")

        elif (abs(action_position_x - action_x_prev) < 5 and abs(action_position_y - action_y_prev) < 5):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is too close to the current position, skipping navigation')
            invalid_goal = True

        else:
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
                nav2_future = self.nav2_client.send_goal_async(nav2_goal_msg)
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
            # Determine how many new pixels were explored
            old_pixels = self.visited_pixels.copy()
            self.visited_pixels = (map_2d_array >= 0).astype(int)
            new_pixels = self.visited_pixels - old_pixels

            # Compute the reward elements (number of new pixels mapped - penalty for travel distance)
            reward_pos = new_pixels/8000
            reward_neg = np.linalg.norm(np.array([robot_pose_in_action[0], robot_pose_in_action[1]]) - np.array([action[0].item(), action[1].item()]))/2
            
            # If it didn't explore enough pixels force a negative reward
            if (new_pixels < 50):
                reward = -0.5
            else:
                reward = max(min(1,reward_pos - reward_neg),0)

            # Check if the goal is reached
            done = bool(np.sum(self.visited_pixels) > COMPLETION_PERCENTAGE*self.max_pixels)
            #print(f"Reward Components - Reward: {reward_pos} Penalty: {reward_neg}")
            print(f'Reward: {reward:.3f}  Done: {done} - visited {np.sum(new_pixels)} new pixels making a total of {np.sum(new_pixels)/self.max_free_pixels*100:.1f}% of mappable pixels')
            
        # Update the observation (occupancy grid map and robot pose)
        obs = self._get_obs()

        # Save the actions and the reward
        with open(LEARNING_RECORD_PATH , mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([action_position_x, action_position_y, reward])

        return obs, reward, done, {}

    # Required for gym environment
    def render(self, mode="human"):
        print('render function')

    # Required for gym environment
    def close(self):
        print('close')

    # Return current observation
    def _get_obs(self):
        return {'map': self.map_msg_to_obs_space(self.map), 'robot_pose': self.map_frame_to_action_space(self.robot_pose)}

    # Conversion Functions

    def map_msg_to_obs_space(self, map_msg, squeeze=False):
        if (map_msg is None):
            return None
        # Convert the occupancy grid map message to a 2d numpy array
        map_2d_array = np.array(map_msg.data, dtype=np.int8).reshape(map_msg.info.height, map_msg.info.width)
        padded_map = np.full((AI_MAP_DIM*MAP_MAX_POOLING, AI_MAP_DIM*MAP_MAX_POOLING), -1, dtype=np.int8)
        padded_map[0:map_2d_array.shape[0],0:map_2d_array.shape[1]] = map_2d_array
        
        # Resample the map with max pooling and then pad it with -1's to the full size
        obs_space_map = skimage.measure.block_reduce(padded_map,(MAP_MAX_POOLING,MAP_MAX_POOLING), np.max)
        if (squeeze):
            return obs_space_map
        
        return np.expand_dims(obs_space_map.astype(np.uint8),axis=2)

    def map_frame_to_action_space(self, pose_in):
        # Going from actual map frame to the AI Obs / Action format
        x = (2*(pose_in.pose.pose.position.x - self.map.info.origin.position.x)/(self.map.info.resolution*MAP_MAX_POOLING*(AI_MAP_DIM-1))) - 1
        y = (2*(pose_in.pose.pose.position.y - self.map.info.origin.position.y)/(self.map.info.resolution*MAP_MAX_POOLING*(AI_MAP_DIM-1))) - 1
        return np.array([x, y], dtype=np.float32)
    
    def action_space_to_map_frame(self, action):
        pose_out = PoseStamped()
        pose_out.header.frame_id = 'map'
        pose_out.pose.position.x = ((action[0].item()+1)*(AI_MAP_DIM-1)*self.map.info.resolution*MAP_MAX_POOLING/2.0) + self.map.info.origin.position.x
        pose_out.pose.position.y = ((action[1].item()+1)*(AI_MAP_DIM-1)*self.map.info.resolution*MAP_MAX_POOLING/2.0) + self.map.info.origin.position.y
        pose_out.pose.orientation.z = 0.0 + self.map.info.origin.orientation.z
        return pose_out

    # Callbacks

    def map_callback(self, map_msg):
        self.map = map_msg
        self.map_reset_status = False
        if (self.visited_pixels is None):
            self.visited_pixels = self.map_msg_to_obs_space(self.map, squeeze=True)

    def costmap_callback(self, costmap_msg):
        self.costmap = costmap_msg
        self.costmap_reset_status = False

    def pose_callback(self, pose_msg):
        self.robot_pose = pose_msg
        self.pose_reset_status = False

    def __del__(self):
        # Cleanup the ROS node
        self.destroy_node()