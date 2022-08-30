from collections import deque
import random
import numpy as np

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        
    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))
    
    # implement (p_i)^a / sumAll(p_i)
    def get_probabilities(self, priority_scale): # stays internal - used for sampling prob
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probs = scaled_priorities / sum(scaled_priorities)
        return sample_probs 
        
    def get_importance(self, probs): # is returned and used
        importance = 1/len(self.buffer) * 1/probs
        importance_normalized = importance / max(importance)
        return importance_normalized
    
    def set_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size) # sample size
        sample_probs = self.get_probabilities(priority_scale) # get probs of whole buffer
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices] # sample from highest-weight indices
        importance = self.get_importance(sample_probs[sample_indices]) # get importance of all samples
        return samples, importance, sample_indices

    def length(self):
        return len(self.buffer)
    
    def get_contents(self):
        contents = []
        for experience in self.buffer:
            contents.append(experience.get_info())
        return contents
    
    def clear(self):
        self.buffer.clear()
        self.priorities.clear()

class Experience():
  def __init__(self, state, action, reward, next_state, done):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done
  
  def get_info(self):
    return self.state, self.action, self.reward, self.next_state, self.done