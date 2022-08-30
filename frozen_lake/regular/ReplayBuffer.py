from collections import deque
import random
import numpy as np

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, experience):
        self.buffer.append(experience)    

    def sample(self, batch_size):
        samples = random.choices(self.buffer, k=batch_size)
        return samples

    def length(self):
        return len(self.buffer)
    
    def get_contents(self):
        contents = []
        for experience in self.buffer:
            contents.append(experience.get_info())
        return contents
    
    def clear(self):
        self.buffer.clear()

class Experience():
  def __init__(self, state, action, reward, next_state, done):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done
  
  def get_info(self):
    return self.state, self.action, self.reward, self.next_state, self.done