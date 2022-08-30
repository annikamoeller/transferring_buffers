import pandas as pd
import os 
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from numpy import loadtxt

def make_experiment_directory(folder_name):
    if os.path.exists(folder_name):
        print("folder name already exists, please use another one")
        sys.exit()
    else:
        os.mkdir(folder_name)    

def write_progress(avg_reward_list, path):
    with open(path, 'a') as progress_file:
        progress_file.write("episode;reward\n")
        for episode, reward in np.ndenumerate(avg_reward_list):
            progress_file.write(f"{episode[0]};{reward}\n")

def inject_buffer(agent, path):
    if os.path.exists(path):
        with open(path, 'rb') as b:
            buffer = pickle.load(b)
            agent.buffer = buffer

def save_buffer(agent, path):
    with open(path, 'wb') as f:
        pickle.dump(agent.buffer, f)

def write_hyperparams(path, env, start_epsilon, min_epsilon, gamma, alpha, 
    buffer_maxlen, episodes, reps, batch_size, decay_rate):
    with open(path, 'a') as f:
        f.write(f"environment: {env} \n \
                epsilon: {start_epsilon} \n \
                min_epsilon: {min_epsilon} \n \
                gamma: {gamma} \n \
                alpha: {alpha} \n \
                buffer_maxlen: {buffer_maxlen} \n \
                episodes: {episodes} \n \
                repetitions: {reps} \n \
                batch_size: {batch_size} \n \
                decay_rate: {decay_rate}")

def one_hot(state, state_space):
    state_arr =np.zeros((1,state_space))
    state_arr[0][state]=1
    return state_arr

def plot(progress_path, plot_path):
    data = pd.read_csv(progress_path, sep=';')
    plt.figure(figsize=(20,15))
    plt.plot(data['reward'])
    plt.title('Cumulative reward', fontsize=22)
    plt.xlabel('Episode', fontsize=18)
    plt.ylabel('Reward', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(['Reward'], loc='upper left', fontsize=18)
    plt.savefig(plot_path)




