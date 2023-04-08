import rclpy
from cas726_project.nav_env import *

from os import makedirs
from rclpy.node import Node

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.env_checker import check_env

class AI_Explorer(Node):
    def __init__(self, load_path=None):
        # Initialize ROS node
        super().__init__('ai_explorer')

        # Create a custom environment for the agent
        self.env = NavEnv()

        # Create logging folder
        makedirs(LOG_DIR, exist_ok=True)
        makedirs(LOG_DIR_TENSORBOARD, exist_ok=True)

        # Set up an evaluation environment for the agent
        self.env = Monitor(self.env, filename=LOG_DIR)
        check_env(self.env)
        self.model = None
        if (load_path == None):
            # Create the RL Agent
            # Define the neural network architecture for the agent
            self.model = TD3("MultiInputPolicy", self.env, verbose=1, buffer_size=10000, learning_starts=20, tensorboard_log=LOG_DIR_TENSORBOARD)
        else:
            self.model = TD3.load(load_path)
            self.model.set_env(self.env)
        # Set up a callback to save model checkpoints during training
        self.checkpoint_callback = CheckpointCallback(save_freq=10, save_path=MODEL_CHECKPOINT_SAVE_PATH, name_prefix='model')
        print('TD3 Agent initialized')

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
        # Update the agent's policy using the TD3 algorithm
        self.model.learn(total_timesteps=int(5), callback=self.checkpoint_callback)
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

