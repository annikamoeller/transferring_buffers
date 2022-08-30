from train_test import train_lander, test_lander
from DQN import DQN
import argparse
import gym
from Utils import *
from gym.envs.registration import register

gravity=-5
register(id='lunarlander-v4', entry_point='gym.envs.box2d:LunarLander', kwargs={'gravity': gravity} )
print("For help run this program with flag -h \n")

# parse command line arguments
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--run_type', type=str, required=True, help = 'Type "test" or "train"')
parser.add_argument('--folder_name', type=str, required=True, help = "Enter unique folder name")
args = parser.parse_args()
folder_name = args.folder_name
run_type = args.run_type 

# agent params
#2env = gym.make('lunarlander-v4')
env = gym.make("lunarlander-v4")
state_space = env.observation_space.shape[0] #states
action_space = env.action_space.n # actions
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.995 # per episode
buffer_maxlen = 100000
#reg_factor = 0.001

# training params
batch_size = 128
training_start = 256 # which step to start training
target_update_freq = 1000
max_episodes = 500
max_steps = 500
reps = 5
train_freq = 4
backup_freq = 100

buffer_path = 'per_gravity_10_5runs/buffer_4.pkl'
hyperparams_path = f"{folder_name}/hyperparams.csv"

# testing params
model_path = 'checkpoints/model_900.h5'
test_max_episodes = 10
test_max_steps = 500
render_lander = False

make_experiment_directory(folder_name)

write_hyperparams(hyperparams_path, env, epsilon, min_epsilon, gamma, learning_rate, buffer_maxlen,
    max_episodes, reps, batch_size, decay_rate, gravity)

ddqn_agent = DQN(state_space, action_space, learning_rate, 
    gamma, epsilon, min_epsilon, decay_rate, buffer_maxlen, network_type='ddqn')

dqn_agent = DQN(state_space, action_space, learning_rate, 
    gamma, epsilon, min_epsilon, decay_rate, buffer_maxlen, network_type='dqn')
    
if run_type == "train":
    train_lander(ddqn_agent, env, batch_size, training_start, 
        target_update_freq, max_episodes, max_steps, reps, train_freq, backup_freq, folder_name, buffer_path)

if run_type == "test":
    test_lander(model_path, test_max_episodes, test_max_steps, render_lander)