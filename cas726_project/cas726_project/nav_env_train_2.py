import rclpy
import numpy as np
import sys
import random
from copy import copy

import gym
from gym import spaces
from rclpy.action import ActionClient
import csv

from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool

import skimage.measure

from robot_sim.srv import LoadMap, ResetMap, SetRobotPose
from robot_sim.action import NavigateToPose

MAP_MAX_POOLING = 2
AI_MAP_DIM = 256
LOG_DIR = './td3_project_monitor'
MAP_PATH = ['./maps/map-small-0.bmp',
            './maps/map-small-1.bmp',
            './maps/map-small-2.bmp',
            './maps/map-small-3.bmp',
            './maps/map-small-4.bmp',
            './maps/map-small-5.bmp',
            './maps/map-small-6.bmp',
            './maps/map-small-7.bmp',
            './maps/map-small-8.bmp',
            './maps/map-small-9.bmp',
            './maps/map-small-10.bmp',
            './maps/map-small-11.bmp']
MAP_LIM_PATH = ['./maps/map-small-1.bmp',
            './maps/map-small-5.bmp',
            './maps/map-small-10.bmp']

RANDOM_MAP = True
MAP_NUM = 1

COMPLETION_PERCENTAGE = 0.95 # of known full map
LEARNING_RECORD_PATH = f'{LOG_DIR}/training-record.csv'

# Custom Gym environment for navigation task
class NavEnvTrain(gym.Env, Node):

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # Initialize Node
        super().__init__('nav_env_train')

        # Define the observation and action space for the environment
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, AI_MAP_DIM, AI_MAP_DIM), dtype=np.uint8) # 255 represents unknown space in order to satisfy uint8
        
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
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseStamped, '/pose', self.pose_callback, 1)
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, '/costmap', self.costmap_callback, 1)
        self.sim_nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.sim_load_map_client = self.create_client(LoadMap, 'load_map')
        self.sim_reset_map_client = self.create_client(ResetMap, 'reset_map')
        self.sim_set_pose_client = self.create_client(SetRobotPose, 'set_robot_pose')
        self.evaluation_publisher = self.create_publisher(Bool, '/evaluation', 1)

        # Wait for simulator to fully activate
        print('Waiting for simulation services')
        self.sim_nav_client.wait_for_server()
        self.sim_load_map_client.wait_for_service()
        self.sim_reset_map_client.wait_for_service()
        self.sim_set_pose_client.wait_for_service()

        # Load the map into the simulator
        self.load_map(MAP_NUM)

        self.file_name = f'ai_explorer_2_action_record_map_rand-{round(COMPLETION_PERCENTAGE*100)}-sac'

        with open(str(self.file_name), mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['Action X', 'Action Y', 'Reward'])

        print('Environment Node initialized')

    def load_map(self, map_num):
        # Load the map into the simulator
        req = LoadMap.Request()
        req.map_path = MAP_LIM_PATH[map_num]
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


    def reset(self):
        
        if RANDOM_MAP:
            ran_num = random.randint(0,2)
            self.load_map(ran_num)
        
        else:
            # Reset the map in the simulator
            req = ResetMap.Request()
            req.keep_robot = False
            future = self.sim_reset_map_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            res = future.result()
            if (not res.success):
                print('Failed to reset map')
                sys.exit(1)

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

        rclpy.spin_once(self)
        
        # Reset the environment to its initial state and return an observation
        self.map = None
        self.costmap = None
        self.visited_pixels = None
        self.robot_pose = None
        self.map_reset_status = True
        self.pose_reset_status = True
        self.costmap_reset_status = True
        print('Waiting for reset')
        while (self.map_reset_status or self.pose_reset_status or self.costmap_reset_status):
            rclpy.spin_once(self)
        print('Environment reset')
        return self.map_msg_to_obs_space()

    def step(self, action):
        rclpy.spin_once(self)

        # Flag to represent whether the robot can reach the goal given by the model
        invalid_goal = False
        
        # Save current pose before moving
        past_action = self.map_frame_to_action_space(self.robot_pose)

        # Calculate the next goal pose
        goal_pose = self.action_space_to_map_frame(action)

        while ((self.map.info.width != self.costmap.info.width) or (self.map.info.height != self.costmap.info.height)):
            rclpy.spin_once(self)

        (map_x, map_y) = self.action_space_to_map_coords(action)
        map_data_index = map_y*self.map.info.width + map_x

        # Determine if the robot should move to the goal
        if ((map_x < 0) or (map_x >= self.map.info.width) or (map_y < 0) or (map_y >= self.map.info.height)):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is outside map, skipping navigation')
            invalid_goal = True

        elif (self.map.data[map_data_index] == -1):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is unknown space, skipping navigation')
            invalid_goal = True
            
        elif (self.map.data[map_data_index] >= 50):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is occupied space, skipping navigation')
            invalid_goal = True

        elif (self.costmap.data[map_data_index] >= 20):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is near occupied space, skipping navigation')
            invalid_goal = True

        elif (abs(goal_pose.pose.position.x - self.robot_pose.pose.position.x) < 0.05 and abs(goal_pose.pose.position.y - self.robot_pose.pose.position.y) < 0.05):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is too close to the current position, skipping navigation')
            invalid_goal = True

        else:
            print(f'Navigating to pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f}')
            # Send the goal pose to the navigation stack
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = copy(goal_pose.pose)
            goal_msg.speed = 1000.0 # m/s
            future = self.sim_nav_client.send_goal_async(goal_msg)

            # Wait for the navigation to complete before taking the next step
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            result_future = goal_handle.get_result_async()

            # print('Waiting for navigation to complete')
            rclpy.spin_until_future_complete(self,result_future)
            # print('Navigation Completed')
            if (not result_future.result().result.success):
                print('Navigation not successful')
                invalid_goal = True

        rclpy.spin_once(self)
        rclpy.spin_once(self)

        done = False

        # Calculate reward
        if invalid_goal:
            reward = -1
        else:
            # Compute the reward (number of new pixels mapped - penalty for travel distance)
            old_pixels = self.visited_pixels
            self.visited_pixels = np.sum((np.array(self.map.data) == 0).astype(int))
            new_pixels = self.visited_pixels - old_pixels

            reward_pos = new_pixels/12000
            reward_neg = np.linalg.norm(np.array([past_action[0].item(), past_action[1].item()]) - np.array([action[0].item(), action[1].item()]))/2
            # TODO: fix reward multipliers
            
            if (new_pixels < 50):
                reward = -0.5
            else:
                reward = max(min(1,reward_pos - reward_neg),0)

            # Check if the goal is reached
            done = bool(self.visited_pixels > COMPLETION_PERCENTAGE*self.max_free_pixels)
            #print(f"Reward Components - reward: {reward_pos} Penalty: {reward_neg}")
            print(f'Reward: {reward:.3f}  Done: {done} - visited {new_pixels} new pixels making a total of {self.visited_pixels/self.max_free_pixels*100:.1f}% of free pixels')
            # np.savetxt("/home/hluo/Downloads/visited_pixels.csv",np.array(self.visited_pixels, dtype=np.int8).reshape(AI_MAP_DIM, AI_MAP_DIM), delimiter = ",")
            
        # Update the observation (occupancy grid map and robot pose)
        obs = self.map_msg_to_obs_space()

        with open(self.file_name , mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([goal_pose.pose.position.x, goal_pose.pose.position.y, reward])

        return obs, reward, done, {'info':42}

    def render(self, mode="human"):
        print('render function')

    def close(self):
        print('close')

    def map_msg_to_obs_space(self):        
        # Convert the occupancy grid map message to a numpy array
        map_2d_array = np.array(self.map.data, dtype=np.int8).reshape(self.map.info.height, self.map.info.width)
        padded_map = np.full((AI_MAP_DIM*MAP_MAX_POOLING, AI_MAP_DIM*MAP_MAX_POOLING), -1, dtype=np.int8)
        #TODO: Insert check to make sure it isn't too big

        map_corner_x = round(AI_MAP_DIM*MAP_MAX_POOLING/2.0 + self.map.info.origin.position.x/self.map.info.resolution)
        map_corner_y = round(AI_MAP_DIM*MAP_MAX_POOLING/2.0 + self.map.info.origin.position.y/self.map.info.resolution)
        padded_map[map_corner_y:(map_corner_y+map_2d_array.shape[0]),map_corner_x:(map_corner_x+map_2d_array.shape[1])] = map_2d_array
        
        # Resample the map with max pooling
        obs_space_map = skimage.measure.block_reduce(padded_map,block_size=(MAP_MAX_POOLING,MAP_MAX_POOLING), func=np.max)
        return np.expand_dims(obs_space_map.astype(np.uint8),axis=0)

    def map_frame_to_action_space(self, pose_in):
        # Going from actual map frame (m, m) to the AI Obs / Action format (-1 to 1, -1 to 1)
        x = pose_in.pose.position.x / (AI_MAP_DIM*MAP_MAX_POOLING*self.map.info.resolution/2.0)
        y = pose_in.pose.position.y / (AI_MAP_DIM*MAP_MAX_POOLING*self.map.info.resolution/2.0)
        return np.array([x, y], dtype=np.float32)
    
    def action_space_to_map_frame(self, action):
        pose_out = PoseStamped()
        pose_out.header.frame_id = 'map'
        pose_out.pose.position.x = action[0].item() * AI_MAP_DIM*MAP_MAX_POOLING*self.map.info.resolution/2.0
        pose_out.pose.position.y = action[1].item() * AI_MAP_DIM*MAP_MAX_POOLING*self.map.info.resolution/2.0
        return pose_out
    
    def action_space_to_map_coords(self, action):
        coord_x = round(action[0].item() * AI_MAP_DIM*MAP_MAX_POOLING/2.0 - self.map.info.origin.position.x/self.map.info.resolution)
        coord_y = round(action[1].item() * AI_MAP_DIM*MAP_MAX_POOLING/2.0 - self.map.info.origin.position.y/self.map.info.resolution)
        return (coord_x, coord_y)

    # Callbacks

    def map_callback(self, map_msg):
        self.map = map_msg
        self.map_reset_status = False
        if (self.visited_pixels is None):
            self.visited_pixels = np.sum((np.array(map_msg.data) == 0).astype(int))

    def costmap_callback(self, costmap_msg):
        self.costmap = costmap_msg
        self.costmap_reset_status = False

    def pose_callback(self, pose_msg):
        self.robot_pose = pose_msg
        self.pose_reset_status = False

    def __del__(self):
        # Cleanup the ROS node
        self.destroy_node()