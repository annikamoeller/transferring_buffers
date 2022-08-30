
class Experience():
  def __init__(self, state, action, reward, next_state, done):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done
  
  def get_info(self):
    return self.state, self.action, self.reward, self.next_state, self.done