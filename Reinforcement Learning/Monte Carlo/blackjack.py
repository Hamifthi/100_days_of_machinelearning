import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')

def monte_carlo_prediction(policy, env, num_episodes, discount_factor = 1.0):
    policy = policy
    V = np.zeros((22, 11, 2))
    return_s = np.zeros((22, 11, 2))
    for episode in range(num_episodes):
        states = []
        G = []
        state = env.reset()
        while state[0] > 21:
            state = env.reset()
        while True:
            states.append(state)
            if state[0] == policy:
                next_state, reward, complete, _ = env.step(0)
            else:
                next_state, reward, complete, _ = env.step(1)
            G.append(reward)
            if complete == True:
                break
            states.append(next_state)
            
        print(states)
        print(G)
        

monte_carlo_prediction(20, env, 10, discount_factor = 1.0)