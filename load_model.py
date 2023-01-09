import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from keras.models import load_model
from agent import Agent

def play_game():
    #initialize game and agent
    env = PLE(FlappyBird(), frame_skip=2, force_fps=True, display_screen=True)
    env.init()

    agent = Agent()

    #load saved model
    agent.model = load_model("trained_model")

    #play game
    env.reset_game()
    while True:
        score = env.score()

        state = agent.get_state(env.getGameState())

        action_index = agent.get_action(state, env.getActionSet())
        action_value = env.getActionSet()[action_index]

        env.act(action_value)

        if env.game_over():
            print("Score:", score)
            break

if __name__ == "__main__":
    play_game()