import rclpy
import numpy as np
import sys
import random

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
            './maps/map-small-11.bmp',]

RANDOM_MAP = False
MAP_NUM = 5

COMPLETION_PERCENTAGE = 0.9 # of known full map
LEARNING_RECORD_PATH = f'{LOG_DIR}/training-record.csv'

# Custom Gym environment for navigation task
class NavEnvTrain(gym.Env, Node):

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # Initialize Node
        super().__init__('nav_env_train')

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

        self.file_name = f'ai_explorer_action_record_map{MAP_NUM}-{round(COMPLETION_PERCENTAGE*100)}-model5'

        with open(str(self.file_name), mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['Action X', 'Action Y', 'Reward'])

        print('Environment Node initialized')

    def load_map(self, map_num):
        # Load the map into the simulator
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
        
        self.max_free_pixels = self.determineDoneCondition(res.map)

    def determineDoneCondition(self, map):
        max_pixels = round(np.sum(np.array(map.data) == 0))
        #print(f'Max number of pixels to map is {max_pixels} or {max_pixels*100/round(len(map.data)/(MAP_MAX_POOLING*MAP_MAX_POOLING)):.2f}% of total pixels')
        return max_pixels

    def isReady(self):
        # Verify that all of the messages have been received since 
        return (not self.map_reset_status and not self.pose_reset_status and not self.costmap_reset_status)

    def reset(self):
        
        if RANDOM_MAP:
            ran_num = random.randint(0,9)

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
        rclpy.spin_once(self)

        # Save pose before moving
        robot_pose_in_action = self.map_frame_to_action_space(self.robot_pose)
        action_x_prev = round((robot_pose_in_action[0].item()+1)*(AI_MAP_DIM-1)/2.0)
        action_y_prev = round((robot_pose_in_action[1].item()+1)*(AI_MAP_DIM-1)/2.0)

        # Flag to represent whether the robot can reach the goal given by the model
        invalid_goal = False
        
        # Set the next goal pose for Nav2
        goal_pose = self.action_space_to_map_frame(action)

        action_position_x = round((action[0].item()+1)*(AI_MAP_DIM-1)/2.0)
        action_position_y = round((action[1].item()+1)*(AI_MAP_DIM-1)/2.0)

        # Determine if the robot should move to the goal
        # print(f"Goal Pose is {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} aka action {action_x} {action_y}")

        while ((self.map.info.width != self.costmap.info.width) or (self.map.info.height != self.costmap.info.height)):
            rclpy.spin_once(self)

        map_data_index = action_position_y*MAP_MAX_POOLING*self.map.info.width + action_position_x*MAP_MAX_POOLING

        if ((action_position_x*MAP_MAX_POOLING >= self.map.info.width) or (action_position_y*MAP_MAX_POOLING >= self.map.info.height)):
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

        # TODO: what should the threshold be for the costmap to avoid the blue area
        elif (abs(action_position_x - action_x_prev) < 5 and abs(action_position_y - action_y_prev) < 5):
            print(f'Pose {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} is too close to the current position, skipping navigation')
            invalid_goal = True

        else:
            # Send the goal pose to the navigation stack
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = goal_pose.pose
            goal_msg.speed = 1.0 # m/s
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

        done = False

        # Calculate reward
        if invalid_goal:
            reward = -1
        else:
            # Compute the reward (number of new pixels mapped - penalty for travel distance)
            old_pixels = self.visited_pixels #TODO: Update visited pixels on the first map / should sit for a couple of seconds and then set this and then start -> maybe handle this in main
            map_data = np.array(self.map.data)
            self.visited_pixels = np.sum((map_data >= 0).astype(int))
            free_pixels = np.sum((map_data == 0).astype(int))
            new_pixels = np.sum(self.visited_pixels - old_pixels)

            reward_pos = new_pixels/8000
            reward_neg = np.linalg.norm(np.array([robot_pose_in_action[0], robot_pose_in_action[1]]) - np.array([action[0].item(), action[1].item()]))/2
            # TODO: fix reward multipliers
            
            if (new_pixels < 50):
                reward = -0.5
            else:
                reward = max(min(1,reward_pos - reward_neg),0)

            # Check if the goal is reached
            # perc_pixels_visited = (np.sum(self.visited_pixels))/(AI_MAP_DIM*AI_MAP_DIM)
            done = bool(free_pixels > COMPLETION_PERCENTAGE*self.max_free_pixels)
            #print(f"Reward Components - reward: {reward_pos} Penalty: {reward_neg}")
            print(f'Reward: {reward:.3f}  Done: {done} - visited {new_pixels} new pixels making a total of {free_pixels/self.max_free_pixels*100:.1f}% of free pixels')
            # np.savetxt("/home/hluo/Downloads/visited_pixels.csv",np.array(self.visited_pixels, dtype=np.int8).reshape(AI_MAP_DIM, AI_MAP_DIM), delimiter = ",")
            
        # Update the observation (occupancy grid map and robot pose)
        obs = self._get_obs()

        with open(self.file_name , mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([action_position_x, action_position_y, reward])

        return obs, reward, done, {'info':42}

    def render(self, mode="human"):
        print('render function')

    def close(self):
        print('close')

    def _get_obs(self):
        return {'map': self.map_msg_to_obs_space(self.map), 'robot_pose': self.map_frame_to_action_space(self.robot_pose)}

    def map_msg_to_obs_space(self, map_msg):
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
        return np.expand_dims(obs_space_map.astype(np.uint8),axis=2)


    def map_frame_to_action_space(self, pose_in):
        # Going from actual map frame (m, m, rad) to the AI Obs / Action format (0-64, 0-64, rad)
        x = (2*(pose_in.pose.position.x - self.map.info.origin.position.x)/(self.map.info.resolution*MAP_MAX_POOLING*(AI_MAP_DIM-1))) - 1
        y = (2*(pose_in.pose.position.y - self.map.info.origin.position.y)/(self.map.info.resolution*MAP_MAX_POOLING*(AI_MAP_DIM-1))) - 1
        return np.array([x, y], dtype=np.float32)
    
    def action_space_to_map_frame(self, action):
        pose_out = PoseStamped()
        pose_out.header.frame_id = 'map'
        pose_out.pose.position.x = ((action[0].item()+1)*(AI_MAP_DIM-1)*self.map.info.resolution*MAP_MAX_POOLING/2.0) + self.map.info.origin.position.x
        pose_out.pose.position.y = ((action[1].item()+1)*(AI_MAP_DIM-1)*self.map.info.resolution*MAP_MAX_POOLING/2.0) + self.map.info.origin.position.y
        return pose_out

    # Callbacks

    def map_callback(self, map_msg):
        # print(f'Map is {map_msg.info.height} by {map_msg.info.width}')
        self.map = map_msg
        self.map_reset_status = False
        if (self.visited_pixels is None):
            self.visited_pixels = np.sum((np.array(map_msg.data) >= 0).astype(int))

    def costmap_callback(self, costmap_msg):
        self.costmap = costmap_msg
        self.costmap_reset_status = False

    def pose_callback(self, pose_msg):
        self.robot_pose = pose_msg
        self.pose_reset_status = False

    def __del__(self):
        # Cleanup the ROS node
        self.destroy_node()