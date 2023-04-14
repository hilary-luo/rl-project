import rclpy
from cas726_project.nav_env import *
from cas726_project.nav_env_train import *

from os import makedirs
from rclpy.node import Node

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common.env_checker import check_env

LOG_DIR = './td3_project_monitor'
LOG_DIR_TENSORBOARD = './td3_project_tensorboard'
MODEL_CHECKPOINT_SAVE_PATH = './td3_nav_checkpoint'
MODEL_SAVE_PATH = 'td3_nav_model'
MODEL_LOAD_PATH = 'td3_nav_model'

class AI_Explorer(Node):
    def __init__(self, load_path=None):
        # Initialize ROS node
        super().__init__('ai_explorer')

        # Create a custom environment for the agent
        #self.env = NavEnv()
        self.env = NavEnvTrain()

        # Create logging folder
        makedirs(LOG_DIR, exist_ok=True)
        makedirs(LOG_DIR_TENSORBOARD, exist_ok=True)

        # Set up an evaluation environment for the agent
        self.env = Monitor(self.env, filename=LOG_DIR)
        check_env(self.env)
        self.model = None
        if (load_path == None):
            # Add some action noise for exploration
            n_actions = self.env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

            # Create the RL Agent
            # Define the neural network architecture for the agent
            #self.model = PPO("MultiInputPolicy", self.env, verbose=2, buffer_size=10000, learning_rate=0.0001, learning_starts=100, batch_size = 10, 
            #                 train_freq=(100,'step'), action_noise=action_noise, seed=12, tensorboard_log=LOG_DIR_TENSORBOARD)
            self.model = PPO("MultiInputPolicy", self.env, verbose=2, batch_size = 32, n_steps=512, seed=12, tensorboard_log=LOG_DIR_TENSORBOARD)
        else:
            self.model = PPO.load(load_path)
            self.model.set_env(self.env)
        # Set up a callback to save model checkpoints during training
        self.checkpoint_callback = CheckpointCallback(save_freq=100, save_path=MODEL_CHECKPOINT_SAVE_PATH, name_prefix='model')
        print('PPO Agent initialized')

        print('AI Explorer Node initialized')

    def predict(self, state):
        # Predict an action using the agent's neural network
        #print('State - Map:')
        #print(state['map'])
        #print('State - Pose:', state['robot_pose'])
        action, _ = self.model.predict(state)
        print('Action:', action)
        return action

    def learn(self, save_path):
        # Update the agent's policy using the PPO algorithm
        self.model.learn(total_timesteps=int(12000), callback=self.checkpoint_callback)
        print('Completed Learn')
        # Save the updated model
        self.model.save(save_path)

def main(args=None):
    rclpy.init(args=args)

    #ai_explorer = AI_Explorer()
    ai_explorer = AI_Explorer(MODEL_LOAD_PATH)

    ai_explorer.learn(MODEL_SAVE_PATH)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ai_explorer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

