import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from agent import Agent

if __name__ == "__main__":
    #parameters
    frame_skip = 2
    force_fps = True #False - slower speed
    display_screen = True

    episodes = 200
    max_score = 0

    #initialize game and agent
    game = FlappyBird()
    env = PLE(game, frame_skip=frame_skip, force_fps=force_fps, display_screen=display_screen)
    env.init()

    agent = Agent(env.getActionSet())

    #start training
    for episode in range(episodes):
        env.reset_game()
        total_reward = 0

        while True:
            action = agent.pick_action()
            reward = env.act(action)
            current_state = np.array(env.getScreenRGB())

            if env.score() > max_score:
                max_score = env.score()

            if env.game_over():
                break
        
    #best result
    print(max_score)