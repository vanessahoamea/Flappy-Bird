import numpy as np

class Agent():
    def __init__(self, actions):
        self.actions = actions

    def pick_action(self):
        return self.actions[np.random.randint(0, len(self.actions))]