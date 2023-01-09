import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow import convert_to_tensor, expand_dims

class Agent:
    def __init__(self):
        self.epsilon = 0
        self.gamma = 0.9
        self.learning_rate = 0.01
        self.model = self.create_model()
        self.memory = deque(maxlen=1000)
        self.batch_size = 250
        self.games_played = 0
    
    def create_model(self):
        model = Sequential()

        model.add(Dense(64, activation="relu", input_shape=(6,)))
        model.add(Dense(2, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(self.learning_rate))

        return model
    
    def get_action(self, state, actions):
        if random.randint(0, 200) < self.epsilon:
        # if np.random.binomial(1, self.epsilon) == 1:
            #select a random action
            action_index = np.random.randint(0, len(actions))
        else:
            #select the optimal action
            state_tensor = expand_dims(convert_to_tensor(state), 0)
            model_output = self.model.predict(state_tensor, verbose=0)
            action_index = np.argmax(model_output[0])
        
        return action_index
    
    def get_state(self, state):
        player_y = state["player_y"]
        horizontal_distance = state["next_pipe_dist_to_player"]
        top_distance = state["next_pipe_top_y"] - state["player_y"]
        bottom_distance = state["next_pipe_bottom_y"] - state["player_y"]
        next_top_distance = state["next_next_pipe_top_y"] - state["player_y"]
        next_bottom_distance = state["next_next_pipe_bottom_y"] - state["player_y"]

        return np.array([player_y, horizontal_distance, top_distance, bottom_distance, next_top_distance, next_bottom_distance])
        
    def experience_replay(self):
        #perform training based on 250 random past experiences
        if len(self.memory) > self.batch_size:
            sample = random.sample(self.memory, self.batch_size)
        else:
            sample = self.memory
        
        #fit model with the calculated Q-values as the target
        input_tensor = []
        target_tensor = []
        for state, action_index, reward, next_state, done in sample:
            target = reward

            if not done:
                next_state_tensor = expand_dims(convert_to_tensor(next_state), 0)
                model_output = self.model.predict(next_state_tensor, verbose=0)
                q_max_next = np.amax(model_output[0])

                target += self.gamma * q_max_next
            
            state_tensor = expand_dims(convert_to_tensor(state), 0)
            target_state = self.model.predict(state_tensor, verbose=0)
            target_state[0][action_index] = target

            input_tensor.append(state_tensor[0])
            target_tensor.append(target_state[0])
        
        self.model.fit(np.array(input_tensor), np.array(target_tensor), epochs=20, verbose=0)