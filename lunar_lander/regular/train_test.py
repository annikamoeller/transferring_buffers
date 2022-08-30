import gym
from ReplayBuffer import *
from Utils import *
from DQN import DQN
import numpy as np
from tensorflow.keras.models import load_model
from numpy import savetxt
from Utils import *

# training function
def train_lander(agent, env, batch_size, training_start, target_update_freq, 
  max_episodes, max_steps, reps, train_freq, backup_freq, folder_name, buffer_path):
  network_type = agent.network_type

  step_counter = 0

  for rep in range(reps):
    agent.reset()
    if buffer_path is not None:
      inject_buffer(agent, buffer_path)
      print(agent.buffer.length())
    avg_reward_deque = deque(maxlen=100)
    reward_tracker = []
    avg_reward_tracker = []
    for episode in range(max_episodes): # training loop
      state = env.reset() 

      episode_reward = 0 # reward tracker

      for step in range(max_steps): # limit number of steps
        step_counter += 1
        action = agent.select_action(state) # get action 
        #env.render()
        next_state, reward, done, info = env.step(action) # next step
        episode_reward += reward # increment reward
        
        if step == max_steps: # stop at max steps 
          print(f"Episode reached the maximum number of steps. {max_steps}")
          done = True

        experience = Experience(state, action, reward, next_state, done) # create new experience object
        agent.buffer.add(experience) # add experience to buffer

        state = next_state # update state

        if network_type is "ddqn":
          if step_counter % target_update_freq == 0: # update target weights every x steps 
            print("Updating target model step: ", step)
            agent.update_target_weights()
          
        if (agent.buffer.length() >= training_start) & (step % train_freq == 0): # train agent every y steps
          batch = agent.buffer.sample(batch_size)

          if network_type is "ddqn":
            inputs, targets = agent.calculate_inputs_and_targets_ddqn(batch)
          else:
            inputs, targets = agent.calculate_inputs_and_targets_dqn(batch)
          agent.train(inputs, targets)

        if done: # stop if this action results in goal reached
          break

      reward_tracker.append(episode_reward)
      avg_reward_deque.append(episode_reward)
      average = np.mean(avg_reward_deque)
      avg_reward_tracker.append(average)

      print(f"EPISODE {episode} finished in {step} steps, " )
      print(f"epsilon {agent.current_epsilon}, reward {episode_reward}. ")
      print(f"Average reward over last 100: {average} \n")
      # if episode != 0 and episode % backup_freq == 0: # back up model every z steps 
      #   backup_model(agent.model, episode, network_type)
      
      agent.epsilon_decay()

    savetxt(f'{folder_name}/progress_{rep}', reward_tracker)
    savetxt(f'{folder_name}/avg_reward_{rep}', avg_reward_tracker)
    save_buffer(agent, f'{folder_name}/buffer_{rep}.pkl')
    backup_model(agent.model, folder_name, rep)
  
# testing function
def test_lander(model_filename, max_episodes, max_steps, render=False):
  env = gym.make("LunarLander-v2")
  trained_model = load_model(model_filename)

  def get_q_values(model, state):
      state = np.array(state)
      state = np.reshape(state, [1, 8])
      return model.predict(state)

  def select_best_action(q_values):
      return np.argmax(q_values)

  rewards = []
  for episode in range(max_episodes):
      state = env.reset()

      episode_reward = 0

      for step in range(1, max_steps+1):
          if render:
            env.render()

          q_values = get_q_values(trained_model, state)
          action = select_best_action(q_values)
          next_state, reward, done, info = env.step(action)

          episode_reward += reward

          if step == max_steps:
              print(f"Episode reached the maximum number of steps. {max_steps}")
              done = True

          state = next_state

          if done:
              break

      print(f"episode {episode} finished in {step} steps with reward {episode_reward}.")
      rewards.append(episode_reward)

  print("Average reward: ", np.average(rewards))