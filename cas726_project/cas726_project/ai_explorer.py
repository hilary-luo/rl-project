import numpy as np
import rclpy

from time import sleep
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import Costmap

import gym
from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import skimage.measure

MAP_MAX_POOLING = 2
AI_MAP_DIM = 256
ACTION_RESOLUTION_FACTOR = 4
AI_OUTPUT_POS_DIM = AI_MAP_DIM/ACTION_RESOLUTION_FACTOR

# Custom Gym environment for navigation task
class NavEnv(gym.Env, Node):
    def __init__(self):
        # Initialize Node
        super().__init__('nav_env')

        # Define the observation and action space for the environment
        self.observation_space = spaces.Dict({
            'map': spaces.Box(low=-1, high=100, shape=(AI_MAP_DIM, AI_MAP_DIM, 1), dtype=np.int8),
            'robot_pose': spaces.Box(low=0, high=AI_OUTPUT_POS_DIM-1, shape=(2,), dtype=np.uint8),
        })

        # AI output will be an xy position on the map with a resolution of every 4 pixels
        self.action_space = spaces.Box(low=0, high=AI_OUTPUT_POS_DIM-1, shape=(2,), dtype=np.uint8)

        # Initialize environment variables
        self.reset()

        # Set up publishers, subscribers and clients
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.pose_subscriber = self.create_subscription(Pose, '/pose', self.pose_callback, 1)
        self.pose_subscriber = self.create_subscription(Costmap, '/global_costmap/costmap', self.costmap_callback, 1)

        # Wait for navigation to fully activate
        print('Waiting for Nav2 server')
        self.nav2_client.wait_for_server()

        print('Environment Node initialized')

    def isReady(self):
        return not self.map_reset_status and not self.pose_reset_status and not self.costmap_reset_status

    def reset(self):
        # Reset the environment to its initial state and return an observation
        self.map = None
        self.costmap = np.full((AI_MAP_DIM, AI_MAP_DIM), -1, dtype=np.int8)
        self.visited_pixels = np.zeros((AI_MAP_DIM, AI_MAP_DIM), dtype=np.int8)
        self.robot_pose = PoseStamped()
        self.robot_pose_prev = PoseStamped()
        self.goal_pose = PoseStamped()
        self.map_reset_status = True
        self.pose_reset_status = True
        self.costmap_reset_status = True
        print('Environment reset')

    def step(self, action):
        # Save pose before moving
        self.robot_pose_prev = self.robot_pose

        # Set the next goal pose for Nav2

        invalid_goal = False

        # Translate from action space into map frame
        self.goal_pose = self.action_space_to_map_frame(self, action)
        map_2d_array = self.map_msg_to_obs_space(self.map)

        # TODO: Do I need to check for a particular radius of not unknown
        if (map_2d_array[action[0].item().pose.position.x*ACTION_RESOLUTION_FACTOR][action[1].item().pose.position.y*ACTION_RESOLUTION_FACTOR] == -1):
            print(f'Pose {self.goal_pose.pose.position.x}, {self.goal_pose.pose.position.y} is unknown space, skipping navigation')
            invalid_goal = True

        # TODO: what should the threshold be for the costmap to avoid the blue area
        elif (self.costmap[action[0].item().pose.position.x*ACTION_RESOLUTION_FACTOR][action[1].item().pose.position.y*ACTION_RESOLUTION_FACTOR] >= 70):
            print(f'Pose {self.goal_pose.pose.position.x}, {self.goal_pose.pose.position.y} is occupied space, skipping navigation')
            invalid_goal = True

        else:

            # Send the goal pose to the navigation stack
            print('Sending the pose')
            nav2_goal_msg = NavigateToPose.Goal()
            nav2_goal_msg.pose = self.goal_pose
            nav2_future = self.nav2_client.send_goal_async(nav2_goal_msg, self.nav_callback)

            # Wait for the navigation to complete before taking the next step
            print('Checking goal acceptance')
            rclpy.spin_until_future_complete(self, nav2_future)
            goal_handle = nav2_future.result()

            if not goal_handle.accepted:
                print('Goal was rejected')
                return False # TODO: Handle this in a smart way

            result_future = goal_handle.get_result_async()

            print('Waiting for navigation to complete')
            rclpy.spin_until_future_complete(self,result_future)
            print('Navigation Completed')

        if invalid_goal:
            reward = -1
        else:
            # Compute the reward (number of new pixels mapped - penalty for travel distance)
            old_pixels = self.visited_pixels.copy() #TODO: Update visited pixels on the first map / should sit for a couple of seconds and then set this and then start -> maybe handle this in main
            self.visited_pixels[map_2d_array >= 0] = 1
            new_pixels = self.visited_pixels - old_pixels
            robot_pose_in_action = self.map_frame_to_action_space(self, self.robot_pose)
            reward = np.sum(new_pixels) - 0.1*np.linalg.norm(np.array(robot_pose_in_action[0], robot_pose_in_action[1]) - np.array([self.goal_pose.pose.position.x, self.goal_pose.pose.position.y]))
            # TODO: Add bounds?
        print(f'Reward: {reward}')

        # Check if the goal is reached
        done = np.sum(self.visited_pixels) > (AI_MAP_DIM*AI_MAP_DIM*0.75) 
        print(f'Done: {done}')

        # Update the observation (occupancy grid map and robot pose)
        obs = self.get_obs()
        print(f'Observation: {obs}')

        return obs, reward, done

    def get_obs(self):
        return {'map': self.map_msg_to_obs_space(self.map), 'robot_pose': self.map_frame_to_action_space(self, self.robot_pose)}

    def map_msg_to_obs_space(map_msg):
        print(f'Map is {map_msg.info.height} by {map_msg.info.width}')
        # Convert the occupancy grid map message to a numpy array
        map_2d_array = np.array(map_msg.data, dtype=np.int8).reshape(map_msg.info.height, map_msg.info.width)
        
        # Resample the map with max pooling and then pad it with -1's to the full size
        obs_space_map = np.full((AI_MAP_DIM, AI_MAP_DIM), -1, dtype=np.int8)
        data = skimage.measure.block_reduce(map_2d_array,(MAP_MAX_POOLING,MAP_MAX_POOLING), np.max)
        #TODO: Insert check to make sure it isn't too big
        obs_space_map[0:data.shape[0],0:data.shape[1]] = data
        return obs_space_map


    def map_callback(self, map_msg):
        # print(f'Map is {map_msg.info.height} by {map_msg.info.width}')
        self.map = map_msg
        self.map_reset_status = False

    def costmap_callback(self, costmap_msg):
        map_raw = np.array(costmap_msg.data, dtype=np.int8).reshape(costmap_msg.info.height, costmap_msg.info.width)
        map_temp = np.full((AI_MAP_DIM, AI_MAP_DIM), -1, dtype=np.int8)
        data = skimage.measure.block_reduce(map_raw,(MAP_MAX_POOLING,MAP_MAX_POOLING), np.max)
        #TODO: Insert check to make sure it isn't too big
        map_temp[0:data.shape[0],0:data.shape[1]] = data
        self.costmap = map_temp
        self.costmap_reset_status = False

    def pose_callback(self, pose_msg):
        self.robot_pose = pose_msg
        self.pose_reset_status = False

    def map_frame_to_action_space(self,pose_in):
        # Going from actual map frame (m, m, rad) to the AI Obs / Action format (0-64, 0-64, rad)
        x = int((pose_in.pose.position.x - self.map.info.origin.pose.position.x)/(self.map.info.resolution*MAP_MAX_POOLING*ACTION_RESOLUTION_FACTOR))
        y = int((pose_in.pose.position.y - self.map.info.origin.pose.position.y)/(self.map.info.resolution*MAP_MAX_POOLING*ACTION_RESOLUTION_FACTOR))
        # z = (pose_in.pose.orientation.z - self.map.info.origin.pose.orientation.z)
        return np.array(x, y, dtype=np.uint8)
    
    def action_space_to_map_frame(self, action):
        pose_out = PoseStamped()
        pose_out.header.frame_id = 'map'
        pose_out.pose.position.x = (action[0].item()*self.map.info.resolution*MAP_MAX_POOLING*ACTION_RESOLUTION_FACTOR) + self.map.info.origin.pose.position.x
        pose_out.pose.position.y = (action[1].item()*self.map.info.resolution*MAP_MAX_POOLING*ACTION_RESOLUTION_FACTOR) + self.map.info.origin.pose.position.y
        pose_out.pose.orientation.z = 0.0 + self.map.info.origin.pose.orientation.z
        return pose_out


    def nav_callback(self, msg):
        #print('Nav Callback')
        ''

    def odom_callback(self, msg):
        #print('Odom Callback')
        ''

    def __del__(self):
        # Cleanup the ROS node
        self.destroy_node()
        rclpy.shutdown()


# Define the TD3 agent and its neural network architecture
class TD3Agent:
    def __init__(self, env):
        # Initialize the TD3 agent with the environment
        self.env = env
        # Define the neural network architecture for the agent
        self.model = TD3("MultiInputPolicy", self.env, verbose=1, buffer_size=10000)
        # Set up a callback to save model checkpoints during training
        self.checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./td3_nav_checkpoint', name_prefix='model')
        # Set up an evaluation environment for the agent
        self.eval_env = DummyVecEnv([lambda: self.env])
        print('TD3 Agent initialized')
    
    def select_action(self, state):
        # Predict an action using the agent's neural network
        print('State - Map:')
        print(state['map'])
        print('State - Pose:', state['robot_pose'])
        action, _ = self.model.predict(state)
        print('Action:', action)
        return action
    
    def update_policy(self, states, actions, rewards):
        # Update the agent's policy using the TD3 algorithm
        self.model.learn(total_timesteps=int(1e6))
        # Save the updated model
        self.model.save('td3_nav_model')

class AI_Explorer(Node):
    def __init__(self):
        # Initialize ROS node
        super().__init__('ai_explorer')

        # Create a custom environment for the agent
        self.env = NavEnv()

        # Create the RL Agent
        self.agent = TD3Agent(self.env)

        print('Explorer Node initialized')
    
    def explore(self):

        # Run the exploration loop until done
        self.env.reset() #TODO: Reset the Nav2 map
        done = False
        while rclpy.ok() and not self.env.isReady():
            rclpy.spin_once(self.env)
        obs = self.env.get_obs()
        while not done:
            # Select an action using the TD3 agent's neural network
            print('Selecting an action')
            action = self.agent.select_action(obs)

            # Take a step in the environment using the selected action
            print('Sending the action')
            obs, reward, done = self.env.step(action)

            # Update the policy of the TD3 agent with the observed states, actions and rewards
            self.agent.update_policy(self, obs, action, reward)

def main(args=None):
    rclpy.init(args=args)

    ai_explorer = AI_Explorer()

    while True:
        ai_explorer.explore()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ai_explorer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

