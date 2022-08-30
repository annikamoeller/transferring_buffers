import numpy as np
from collections import deque 
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle

# def backup_model(model, episode, network_type):
#     backup_file = f"checkpoints_{network_type}/model_{episode}.h5"
#     print(f"Backing up model to {backup_file}")
#     model.save(backup_file)

def backup_model(model, folder_name, rep):
    backup_file = f"{folder_name}/model_{rep}.h5"
    model.save(backup_file)

def plot(logger, network_type):
  data = pd.read_csv(logger.file_name, sep=';')
  plt.figure(figsize=(11,10))
  plt.plot(data['average'])
  plt.plot(data['reward'])
  plt.title('Reward per training episode', fontsize=22)
  plt.xlabel('Episode', fontsize=18)
  plt.ylabel('Reward', fontsize=18)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.legend(['Average reward', 'Reward'], loc='upper left', fontsize=18)
  plt.savefig(f'metrics_{network_type}/reward_plot.png')

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
    buffer_maxlen, episodes, reps, batch_size, decay_rate, gravity):
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
                decay_rate: {decay_rate} \n \
                gravity: {gravity} ")
