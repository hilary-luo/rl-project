import rclpy
from cas726_project.nav_env import *
from cas726_project.nav_env_train import *
from cas726_project.evaluator import *

from os import makedirs
from os.path import exists
from rclpy.node import Node
from time import sleep

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

MODEL_TYPE = 'sac' # options are 'sac' or 'ppo'

LOG_DIR_MONITOR = f'./{MODEL_TYPE}_project_monitor'
LOG_DIR_TENSORBOARD = f'./TensorBoard/{MODEL_TYPE}_project_tensorboard'
MODEL_CHECKPOINT_PATH = f'./{MODEL_TYPE}_checkpoint'
TB_LOG_NAME = f'{MODEL_TYPE}_model'
MODEL_SAVE_PATH = f'{MODEL_TYPE}_model'
MODEL_LOAD_PATH = MODEL_SAVE_PATH

USE_CUSTOM_SIM = True
TRAIN_OPTION = False

class AI_Explorer(Node):
    def __init__(self, load_path=None):
        # Initialize ROS node
        super().__init__('ai_explorer')

        # Create a custom environment for the agent based on which sim is being used
        self.env = None
        if USE_CUSTOM_SIM:
            self.env = NavEnvTrain() # For custom simulator
        else:
            self.env = NavEnv() # For native turtlebot simulator

        # Create logging folders
        makedirs(LOG_DIR_MONITOR, exist_ok=True)
        makedirs(LOG_DIR_TENSORBOARD, exist_ok=True)
        makedirs(MODEL_CHECKPOINT_PATH, exist_ok=True)

        # Set up an evaluation environment for the agent
        self.env = Monitor(self.env, filename=LOG_DIR_MONITOR)
        if TRAIN_OPTION:
            check_env(self.env)
        self.model = None
        if (load_path == None):
            # Create the RL model
            if MODEL_TYPE == 'sac':
                self.model = SAC("MultiInputPolicy", self.env, verbose=2, buffer_size=10000, train_freq=(512, "step"), 
                                 batch_size=64, tensorboard_log=LOG_DIR_TENSORBOARD)
            elif MODEL_TYPE == 'ppo':
                self.model = PPO("MultiInputPolicy", self.env, verbose=2, n_steps=512, batch_size=64, seed=12,
                                 tensorboard_log=LOG_DIR_TENSORBOARD)
            else:
                print(f'Invalid model type (must be sac or ppo): {MODEL_TYPE}')
                sys.exit(1)
        else:
            if MODEL_TYPE == 'sac':
                self.model = SAC.load(f'{load_path}.zip')
            elif MODEL_TYPE == 'ppo':
                self.model = PPO.load(f'{load_path}.zip')
            else:
                print(f'Invalid model type (must be sac or ppo): {MODEL_TYPE}')
                sys.exit(1)

            self.model.set_env(self.env)
            print(f'Loaded model: {load_path}')

        # Set up a callback to save model checkpoints during training
        self.checkpoint_callback = CheckpointCallback(save_freq=1024, save_path=MODEL_CHECKPOINT_PATH, name_prefix='model')
        
        print('AI Explorer Node initialized')

    def learn(self, save_path):
        # Update the agent's policy using the algorithm
        self.model.learn(total_timesteps=int(1024*100), callback=self.checkpoint_callback, reset_num_timesteps=False, tb_log_name=TB_LOG_NAME)
        print('Completed Learn')
        # Save the updated model
        self.model.save(save_path)

    def predict(self, state):
        # Predict an action using the agent's neural network
        action, _ = self.model.predict(state)
        print('Action:', action)
        return action
    
    def execute(self, state):
        # Use the model predictions to move the robot
        action = self.predict(state)
        return self.env.step(action)

def main(args=None):
    rclpy.init(args=args)
    ai_explorer = None

    # Check for model to load
    if exists(f'{MODEL_LOAD_PATH}.zip'):
        ai_explorer = AI_Explorer(MODEL_LOAD_PATH)
    else:
        ai_explorer = AI_Explorer()

    # Train the model
    if TRAIN_OPTION:
        ai_explorer.learn(MODEL_SAVE_PATH)

    # Test the model
    else:
        done = False
        state = ai_explorer.env.reset()

        # Tell the evaluator to start recording progress
        eval_status = Bool()
        eval_status.data = True
        ai_explorer.env.evaluation_publisher.publish(eval_status)

        # Use the model to explore until done
        while not done:
            state, _, done, _ = ai_explorer.execute(state)
        
        # Grace period for simulator / evaluator to catch up
        sleep(5)

        # Tell the evaluator to stop recording progress
        eval_status.data = False
        ai_explorer.env.evaluation_publisher.publish(eval_status)

    # Destroy the node explicitly
    ai_explorer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

