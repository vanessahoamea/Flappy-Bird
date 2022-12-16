import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from agent import Agent

if __name__ == "__main__":
    #initialize game and agent
    env = PLE(FlappyBird(), frame_skip=2, force_fps=True, display_screen=True)
    env.init()

    agent = Agent()

    #start training
    max_episodes = 100
    file = open("data/scores.txt", "w")

    for episode in range(max_episodes):
        best_score = 0
        average_score = 0

        #play 100 games before updating the weights
        for attempt in range(100):
            env.reset_game()

            while True:
                score = env.score()

                #get current state
                state = agent.get_state(env.getGameState())

                #choose an action
                action_index = agent.get_action(state, env.getActionSet())
                action_value = env.getActionSet()[action_index]

                #take selected action and move to the next state
                reward = env.act(action_value) * 100
                # reward = agent.get_reward(env)
                next_state = agent.get_state(env.getGameState())
                done = env.game_over()

                #save in replay buffer
                agent.memory.append((state, action_index, reward, next_state, done))

                if done:
                    break
                
            average_score += score
            
            if score > best_score:
                best_score = score

        #save results
        average_score //= 100
        file.write(f"{average_score} {best_score}\n")
        
        print(f"Episode {episode}: Average score: {average_score} \t Best score: {best_score}")

        #train the network
        agent.experience_replay()
        
        #decay probability of taking random action and importance of future rewards
        if agent.epsilon > 0.1:
            agent.epsilon -= 0.008
        if agent.gamma > 0.1:
            agent.gamma -= 0.008

    file.close()

    #save trained model
    agent.model.save("trained_model")