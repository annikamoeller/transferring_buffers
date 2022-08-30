import tensorflow as tf
from keras.optimizers import adam_v2
import numpy as np
from keras.layers import Dense 
from keras.models import Sequential
from ReplayBuffer import ReplayBuffer

class DQN():

    def __init__(self, state_space, action_space, start_epsilon, min_epsilon, 
        gamma, alpha, buffer_maxlen, decay_rate):
        self.action_space = action_space
        self.state_space = state_space
        self.start_epsilon = start_epsilon
        self.current_epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.buffer_maxlen = buffer_maxlen
        self.decay_rate = decay_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.buffer = ReplayBuffer(buffer_maxlen)
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.state_space, input_dim=self.state_space, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=self.alpha))
        return model
    
    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() < self.current_epsilon:
            return np.random.choice(range(self.action_space))
        state = np.reshape(state, [1, self.state_space]) # reshape for .predict()
        q_vals = self.model.predict(state)
        return np.argmax(q_vals[0])
    
    def train(self, experience, batch_size):
        batch = self.buffer.sample(batch_size)
        targets = []
        states = [experience.state[0] for experience in batch]
        next_states = [experience.next_state[0] for experience in batch]

        states_np = np.array(states)
        next_states_np = np.array(next_states)
        q_vals_states = self.model.predict(states_np)
        q_vals_next_states = self.target_model.predict(next_states_np)

        for index, experience in enumerate(batch):
            q_values = q_vals_next_states[index]
            best_action = np.amax(q_values)
            if experience.done:
                target = experience.reward
            else:
                target = experience.reward + (self.gamma * best_action)
            
            target_vector = q_vals_states[index]
            target_vector[experience.action] = target
            targets.append(target_vector)
            
        targets = np.array(targets)
        inputs = np.array(states).reshape(batch_size, self.state_space)
        self.model.fit(inputs, targets, epochs=1, batch_size=batch_size, verbose=0)

    def epsilon_decay(self):
        if self.current_epsilon > self.min_epsilon:
            self.current_epsilon *= self.decay_rate

    def reset(self):
        self.buffer.clear()
        self.current_epsilon = self.start_epsilon
        self.model = self.build_model()