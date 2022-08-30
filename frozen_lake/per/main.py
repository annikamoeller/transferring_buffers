import gym
import numpy as np
from Utils import * 
from Agent import DQN
from ReplayBuffer import Experience
from numpy import savetxt

slip_4x4 = gym.make('FrozenLake-v1', is_slippery=True)
no_slip_4x4 = gym.make('FrozenLake-v1', is_slippery=False)

slip_8x8 = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True)
no_slip_8x8 = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False)

env = slip_8x8
state_space = env.observation_space.n
action_space = env.action_space.n
epsilon = 0.99
min_epsilon = 0.05
gamma = 0.97
alpha = 0.001
decay_rate = 0.99
buffer_maxlen = 1000
episodes = 250
repetitions = 5
batch_size = 50
update_freq = 500
folder_name = 'test'
read_buffer_path = 'no_slip_8x8_buffer/buffer.pkl'
progress_path = f'{folder_name}/progress'
write_buffer_path = f'{folder_name}/buffer.pkl'
plot_path = f'{folder_name}/reward_plot.png'
hyperparams_path = f'{folder_name}/hyperparams.csv'

agent = DQN(state_space, action_space, 
    epsilon, min_epsilon, gamma, alpha, buffer_maxlen, decay_rate)

make_experiment_directory(folder_name)

write_hyperparams(hyperparams_path, env, epsilon, min_epsilon, gamma,
     alpha, buffer_maxlen, episodes, repetitions, batch_size, decay_rate)

for rep in range(repetitions):
    print(f'Repetition: {rep}')
    agent.reset()
    reward_tracker = []
    total_reward = 0
    if read_buffer_path is not None: 
        inject_buffer(agent, read_buffer_path)
        print("buffer injected")
    steps = 0
    for episode in range(episodes):
        state = env.reset()
        state = one_hot(state, state_space)
        done = False
        while not done:
            steps += 1
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = one_hot(next_state, state_space)
            experience = Experience(state, action, reward, next_state, done)
            agent.buffer.add(experience) # add experience
            if (agent.buffer.length() >= batch_size):
                agent.train(experience, batch_size) # train agent
                if steps % update_freq == 0:
                    print("Updating target model step: ", steps)
                    agent.update_target_weights()
            state = next_state
            env.render()
        total_reward += reward
        reward_tracker.append(total_reward)
        agent.epsilon_decay()
        print("Episode: {}, Total reward: {}, Epsilon: {}".format(episode,total_reward, agent.current_epsilon))

    save_buffer(agent, write_buffer_path)
    savetxt(f'{progress_path}_{rep}', reward_tracker)