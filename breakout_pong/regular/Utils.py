import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
import numpy as np
import random

import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys

def write_hyperparams(path, env, start_epsilon, min_epsilon, gamma, alpha, 
    buffer_maxlen, max_steps, batch_size, train_freq, 
    target_update_freq, backup_freq, episodes, repetitions):
    with open(path, 'a') as f:
        f.write(f"environment: {env} \n \
                epsilon: {start_epsilon} \n \
                min_epsilon: {min_epsilon} \n \
                gamma: {gamma} \n \
                alpha: {alpha} \n \
                buffer_maxlen: {buffer_maxlen} \n \
                max_steps: {max_steps} \n \
                batch_size: {batch_size} \n \
                train_freq: {train_freq} \n \
                target_update_freq: {target_update_freq} \n \
                backup_freq: {backup_freq} \n \
                episodes: {episodes} \n \
                repetitions: {repetitions}")

def inject_buffer(agent, path):
    if os.path.exists(path):
        with open(path, 'rb') as b:
            buffer = pickle.load(b)
            agent.buffer = buffer

def save_buffer(agent, path):
    with open(path, 'wb') as f:
        pickle.dump(agent.buffer, f)

def make_experiment_directory(folder_path):
    if os.path.exists(folder_path):
        print("folder name already exists, please use another one")
        sys.exit()
    else:
        os.mkdir(folder_path)
        os.mkdir(f'{folder_path}/buffers')
        os.mkdir(f'{folder_path}/checkpoints')

def backup_model(model, folder_path, episode):
    backup_file = f"{folder_path}/checkpoints/model_{episode}.h5"
    print(f"Backing up model to {backup_file}")
    model.save(backup_file)

def plot(logger):
  data = pd.read_csv(logger.file_name, sep=';')
  plt.figure(figsize=(20,15))
  plt.plot(data['average'])
  plt.plot(data['reward'])
  plt.title('Reward per training episode', fontsize=22)
  plt.xlabel('Episode', fontsize=18)
  plt.ylabel('Reward', fontsize=18)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.legend(['Average reward', 'Reward'], loc='upper left', fontsize=18)
  plt.savefig(f'metrics/reward_plot.png')
  
  
def get_probabilities(priorities, priority_scale): # stays internal - used for sampling prob
    scaled_priorities = np.array(priorities) ** priority_scale
    sample_probs = scaled_priorities / sum(scaled_priorities)
    return sample_probs 

def get_importance(probs): # is returned and used
    importance = 1/len(probs) * 1/probs
    importance_normalized = importance / max(importance)
    return importance_normalized

def sample(self, batch_size, priority_scale=0.6):# was 1.0
    sample_size = min(len(self.buffer), batch_size)
    sample_probs = self.get_probabilities(priority_scale) # get probs of whole buffer
    sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
    samples = np.array(self.buffer)[sample_indices] # sample from highest-weight indices
    importance = self.get_importance(sample_probs[sample_indices]) # get importance of all samples
    return samples, importance, sample_indices
