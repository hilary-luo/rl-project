import rclpy
from cas726_project.nav_env import *
from cas726_project.nav_env_train import *
#from cas726_project.nav_env_train_2 import *
from cas726_project.evaluator import *

from os import makedirs
from os.path import exists
from rclpy.node import Node
from time import sleep

from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common.env_checker import check_env

LOG_DIR = './sac_project_monitor'
LOG_DIR_TENSORBOARD = './TensorBoard/sac_project_tensorboard'
MODEL_CHECKPOINT_SAVE_PATH = './sac_nav_checkpoint'
TB_LOG_NAME = f'SAC-2-April17-map-1'
MODEL_SAVE_PATH = 'sac_nav_model2'
MODEL_LOAD_PATH = 'ppo_nav_model_9'

CUSTOM_SIM_OPTION = True
TRAIN_OPTION = False

class AI_Explorer(Node):
    def __init__(self, load_path=None):
        # Initialize ROS node
        super().__init__('ai_explorer')

        # Create a custom environment for the agent
        self.env = None
        if CUSTOM_SIM_OPTION:
            self.env = NavEnvTrain()
        else:
            self.env = NavEnv()

        # Create logging folder
        makedirs(LOG_DIR, exist_ok=True)
        makedirs(LOG_DIR_TENSORBOARD, exist_ok=True)

        # Set up an evaluation environment for the agent
        self.env = Monitor(self.env, filename=LOG_DIR)
        if TRAIN_OPTION:
            check_env(self.env)
        self.model = None
        if (load_path == None):
            # Add some action noise for exploration
            #n_actions = self.env.action_space.shape[-1]
            #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

            # Create the RL Agent
            # Define the neural network architecture for the agent
            self.model = SAC("CnnPolicy", self.env, verbose=2, buffer_size=10000, train_freq=(1024, "step"), batch_size=64, tensorboard_log=LOG_DIR_TENSORBOARD)
            #self.model = SAC("MultiInputPolicy", self.env, verbose=2, buffer_size=10000, train_freq=(128, "step"), batch_size=64, tensorboard_log=LOG_DIR_TENSORBOARD)
            #self.model = TD3("MultiInputPolicy", self.env, verbose=2, buffer_size=10000, batch_size = 64, train_freq=(256,'step'), action_noise=action_noise, seed=12, tensorboard_log=LOG_DIR_TENSORBOARD)
            #self.model = TD3("MultiInputPolicy", self.env, verbose=2, buffer_size=10000, learning_rate=0.0001, learning_starts=100, batch_size = 10, 
            #                 train_freq=(100,'step'), action_noise=action_noise, seed=12, tensorboard_log=LOG_DIR_TENSORBOARD)
            #self.model = PPO("CnnPolicy", self.env, verbose=2, batch_size = 64, n_steps=1024, seed=12, 
            #                 tensorboard_log=LOG_DIR_TENSORBOARD)
        else:
            print('Loaded model')
            self.model = PPO.load(f'{load_path}.zip')
            #self.model = SAC.load(f'{load_path}.zip')
            self.model.set_env(self.env)
            print(f'Loaded model {load_path}')
        # Set up a callback to save model checkpoints during training
        self.checkpoint_callback = CheckpointCallback(save_freq=1024, save_path=MODEL_CHECKPOINT_SAVE_PATH, name_prefix='model')
        print('Agent initialized')

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
        # Update the agent's policy using the algorithm
        self.model.learn(total_timesteps=int(1024*100), callback=self.checkpoint_callback, reset_num_timesteps=False, tb_log_name=TB_LOG_NAME)
        print('Completed Learn')
        # Save the updated model
        self.model.save(save_path)

    def execute(self, state):
        action = self.predict(state)
        return self.env.step(action)

def main(args=None):
    rclpy.init(args=args)
    ai_explorer = None
    if exists(f'{MODEL_LOAD_PATH}.zip'):
        ai_explorer = AI_Explorer(MODEL_LOAD_PATH)
    else:
        ai_explorer = AI_Explorer()

    if TRAIN_OPTION:
        ai_explorer.learn(MODEL_SAVE_PATH)
    else:
        done = False
        state = ai_explorer.env.reset()

        eval_status = Bool()
        eval_status.data = True
        ai_explorer.env.evaluation_publisher.publish(eval_status)
        while not done:
            state, reward, done, ____ = ai_explorer.execute(state)
            
        sleep(5)
        eval_status.data = False
        ai_explorer.env.evaluation_publisher.publish(eval_status)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ai_explorer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

