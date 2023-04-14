import rclpy
import numpy as np
import sys

import gym
from gym import spaces
from rclpy.action import ActionClient

from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid

import skimage.measure

from robot_sim.srv import LoadMap, ResetMap, SetRobotPose
from robot_sim.action import NavigateToPose

MAP_MAX_POOLING = 2
AI_MAP_DIM = 256
LOG_DIR = './td3_project_monitor'
MAP_PATH = ['./maps/map-simple-1.bmp',
            './maps/map-small.bmp']

COMPLETION_PERCENTAGE = 0.8 # of known full map
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
        self.sim_nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        # self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.pose_subscriber = self.create_subscription(PoseStamped, '/pose', self.pose_callback, 1)
        # self.costmap_subscriber = self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 1)
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, '/costmap', self.costmap_callback, 1)
        self.sim_load_map_client = self.create_client(LoadMap, 'load_map')
        self.sim_reset_map_client = self.create_client(ResetMap, 'reset_map')
        self.sim_set_pose_client = self.create_client(SetRobotPose, 'set_robot_pose')

        # Wait for simulator to fully activate
        print('Waiting for simulation services')
        self.sim_nav_client.wait_for_server()
        self.sim_load_map_client.wait_for_service()
        self.sim_reset_map_client.wait_for_service()
        self.sim_set_pose_client.wait_for_service()

        # Load the map into the simulator
        req = LoadMap.Request()
        req.map_path = MAP_PATH[0]
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

        print('Environment Node initialized')

    def determineDoneCondition(self, map):
        max_pixels = round(np.sum(np.array(map.data) == 0)/(MAP_MAX_POOLING*MAP_MAX_POOLING))
        #print(f'Max number of pixels to map is {max_pixels} or {max_pixels*100/round(len(map.data)/(MAP_MAX_POOLING*MAP_MAX_POOLING)):.2f}% of total pixels')
        return max_pixels

    def isReady(self):
        return (not self.map_reset_status and not self.pose_reset_status and not self.costmap_reset_status)

    def reset(self): #TODO: Reset the Nav2 map
        
        # Load the map into the simulator
        req = ResetMap.Request()
        req.keep_robot = False
        future = self.sim_reset_map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if (not res.success):
            print('Failed to reset map')
            sys.exit(1)
        
        # Load the map into the simulator
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
        while rclpy.ok() and not self.isReady():
            print('waiting for reset')
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
        map_2d_array = self.map_msg_to_obs_space(self.map, squeeze=True)
        costmap_2d_array = self.map_msg_to_obs_space(self.costmap, squeeze=True)
        # np.savetxt("/home/hluo/Downloads/map_obs_space.csv", map_2d_array, delimiter = ",")
        # np.savetxt("/home/hluo/Downloads/costmap_obs_space.csv", costmap_2d_array, delimiter = ",")
        # np.savetxt("/home/hluo/Downloads/map_full.csv",np.array(self.map.data, dtype=np.int8).reshape(self.map.info.height, self.map.info.width), delimiter = ",")
        # np.savetxt("/home/hluo/Downloads/costmap_full.csv",np.array(self.costmap.data, dtype=np.int8).reshape(self.costmap.info.height, self.costmap.info.width), delimiter = ",")

        action_x = round((action[0].item()+1)*(AI_MAP_DIM-1)/2.0)
        action_y = round((action[1].item()+1)*(AI_MAP_DIM-1)/2.0)

        # Determine if the robot should move to the goal
        # print(f"Goal Pose is {goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f} aka action {action_x} {action_y}")

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
            # print(f"Action x: {action_x}  Old position x: {action_x_prev} Action y: {action_y}  Old position y: {action_y_prev}")
            # print(f"Occupancy value: {map_2d_array[action_y][action_x]}    Costmap value {costmap_2d_array[action_y][action_x]}")
            # Send the goal pose to the navigation stack
            #print('Sending the pose')
            nav2_goal_msg = NavigateToPose.Goal()
            nav2_goal_msg.pose = goal_pose.pose
            nav2_goal_msg.speed = 1000.0 # m/s
            nav2_future = self.sim_nav_client.send_goal_async(nav2_goal_msg, self.nav_callback)

            # Wait for the navigation to complete before taking the next step
            #print('Checking goal acceptance')
            rclpy.spin_until_future_complete(self, nav2_future)
            goal_handle = nav2_future.result()

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
            map_2d_array = self.map_msg_to_obs_space(self.map, squeeze=True)
            # Compute the reward (number of new pixels mapped - penalty for travel distance)
            old_pixels = self.visited_pixels.copy() #TODO: Update visited pixels on the first map / should sit for a couple of seconds and then set this and then start -> maybe handle this in main
            self.visited_pixels = (map_2d_array >= 0).astype(int)
            free_pixels = np.sum((map_2d_array == 0).astype(int))
            new_pixels = np.sum(self.visited_pixels - old_pixels)

            reward_pos = new_pixels/2000
            reward_neg = np.linalg.norm(np.array([robot_pose_in_action[0], robot_pose_in_action[1]]) - np.array([action[0].item(), action[1].item()]))/2
            # TODO: fix reward multipliers
            
            if (new_pixels < 50):
                reward = -1
            else:
                reward = max(min(1,reward_pos - reward_neg),0)

            # Check if the goal is reached
            # perc_pixels_visited = (np.sum(self.visited_pixels))/(AI_MAP_DIM*AI_MAP_DIM)
            done = bool(free_pixels > COMPLETION_PERCENTAGE*self.max_free_pixels)
            print(f"Reward Components - reward: {reward_pos} Penalty: {reward_neg}")
            print(f'Reward: {reward:.3f}              Done: {done} - visited {new_pixels} new pixels making a total of {free_pixels/self.max_free_pixels*100:.1f}% of free pixels\n')
            # np.savetxt("/home/hluo/Downloads/visited_pixels.csv",np.array(self.visited_pixels, dtype=np.int8).reshape(AI_MAP_DIM, AI_MAP_DIM), delimiter = ",")
            
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
        x = (2*(pose_in.pose.position.x - self.map.info.origin.position.x)/(self.map.info.resolution*MAP_MAX_POOLING*(AI_MAP_DIM-1))) - 1
        y = (2*(pose_in.pose.position.y - self.map.info.origin.position.y)/(self.map.info.resolution*MAP_MAX_POOLING*(AI_MAP_DIM-1))) - 1
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