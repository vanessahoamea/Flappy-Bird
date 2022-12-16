import numpy as np
import matplotlib.pyplot as plt
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
    trained_model = load_model("trained_model")
    agent.model = trained_model
    agent.epsilon = 0

    #play game
    try:
        env.reset_game()
        while True:

            state = agent.get_state(env.getGameState())

            action_index = agent.get_action(state, env.getActionSet())
            action_value = env.getActionSet()[action_index]

            env.act(action_value)

            if env.game_over():
                break
    except KeyboardInterrupt:
        pass

def plot_convergence():
    file = open("data/scores.txt", "r")

    average_scores = []
    best_scores = []
    for line in file:
        split = line.split()
        average_scores.append(int(float(split[0])))
        best_scores.append(int(float(split[1])))
    
    file.close()

    plt.plot([i+1 for i in range(100)], average_scores)
    plt.plot([i+1 for i in range(100)], best_scores, color="red")
    plt.legend(["average score", "highest score"])
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("data/convergence.png")

if __name__ == "__main__":
    play_game()
    # plot_convergence()