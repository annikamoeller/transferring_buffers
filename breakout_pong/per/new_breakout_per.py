from multiprocessing import reduction
from atari_wrappers import make_atari, wrap_deepmind
from collections import deque
from Utils import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
import argparse
from numpy import savetxt
import random

# parse command line arguments
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--folder_name', type=str, required=True, help = "Enter unique folder name")
args = parser.parse_args()
folder_name = args.folder_name

folder_path = f'/scratch/s3585832/atari_data/per/breakout/{folder_name}'
read_buffer_path = None # 'no_slip_8x8_buffer/buffer.pkl'
progress_path = f'{folder_path}/progress'
avg100_progress_path = f'{folder_path}/avg100_progress'
write_buffer_dir = f'{folder_path}/buffers'
plot_path = f'{folder_path}/reward_plot.png'
hyperparams_path = f'{folder_path}/hyperparams.csv'

reward_tracker = []
avg_reward_deque = deque(maxlen=100)
avg_reward_tracker = []

#make_experiment_directory(folder_name)

# Obseravtions (wrapped)
env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)

num_actions = 4
def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

# The first model makes the predictions for Q-values which are used to make a action.
model = create_q_model()
# Target model
model_target = create_q_model()
model.summary()

gamma = 0.99  # Discount factor for past rewards

# Setting epsilon decay parameters
epsilon = 1.0  
epsilon_max_1 = 1.0 
epsilon_min_1 = 0.2  
epsilon_max_2 = epsilon_min_1  
epsilon_min_2 = 0.1
epsilon_max_3 = epsilon_min_2  
epsilon_min_3 = 0.02

epsilon_interval_1 = (epsilon_max_1 - epsilon_min_1)  
epsilon_interval_2 = (epsilon_max_2 - epsilon_min_2)  
epsilon_interval_3 = (epsilon_max_3 - epsilon_min_3)  

# Number of frames for exploration
epsilon_greedy_frames = 1000000.0

# Number of frames to take random action and observe output
epsilon_random_frames = 50000

# Maximum Replay Buffer volume
max_memory_length = 190000

# Size of batch taken from replay buffer
batch_size = 32  
max_steps_per_episode = 10000

# Train the model after 20 actions
update_after_actions = 20

# How often to update the target network
update_target_network = 10000

#PER offset value
offset = 0.1

# In the Deepmind paper they use RMSProp however then Adam optimizer improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Using huber loss for stability
loss_function = tf.keras.losses.Huber()

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
priorities = []

episode_count = 0
frame_count = 0

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
        # Decay probability of taking random action
        if frame_count < epsilon_greedy_frames:
          epsilon -= epsilon_interval_1 / epsilon_greedy_frames
          epsilon = max(epsilon, epsilon_min_1)
        
        if frame_count > epsilon_greedy_frames and frame_count < 2 * epsilon_greedy_frames:
          epsilon -= epsilon_interval_2 / epsilon_greedy_frames
          epsilon = max(epsilon, epsilon_min_2)
        
        if frame_count > 2 * epsilon_greedy_frames:
          epsilon -= epsilon_interval_3 / epsilon_greedy_frames
          epsilon = max(epsilon, epsilon_min_3)
          

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        priorities.append(max(priorities, default=1))
        state = state_next

        # Update every 20th frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            sample_probs = get_probabilities(priorities, priority_scale=0.6) # get probs of whole buffer
            indices = random.choices(range(len(priorities)), k=batch_size, weights=sample_probs)
            importances = get_importance(sample_probs[indices])
            importances = np.asarray(importances).astype('float32')

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
               [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            updated_q_values = tf.reshape(updated_q_values, [32,1])
            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)
                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                q_action = tf.reshape(q_action, [32,1])
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action, sample_weight=importances**(0.4))

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # TD-error for priority values 
            errors = abs(q_action - updated_q_values)
            errors = errors.numpy()
            for i,e in zip(indices, errors):
                priorities[i] = abs(e) + offset

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]
            del priorities[:1]

        if done:
            break

    reward_tracker.append(episode_reward)
    avg_reward_deque.append(episode_reward)
    avg_last_100 = np.mean(avg_reward_deque) 
    avg_reward_tracker.append(avg_last_100)
    template = "reward: {:.2f}, running reward: {:.2f} at episode {}, frame count {}, epsilon {:.3f}"
    print(template.format(episode_reward, avg_last_100, episode_count, frame_count, epsilon))

    episode_count += 1

    if frame_count >= 10000000:#if avg_last_100 > 18:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        savetxt(progress_path, reward_tracker)
        savetxt(avg100_progress_path, avg_reward_tracker)
        backup_model(model, folder_name, episode_count)
        np.save(f'{write_buffer_dir}/state_history', state_history)
        np.save(f'{write_buffer_dir}/action_history', action_history)
        np.save(f'{write_buffer_dir}/state_next_history', state_next_history)
        np.save(f'{write_buffer_dir}/reward_history', rewards_history)
        np.save(f'{write_buffer_dir}/done_history', done_history)

        break