import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')

def monte_carlo_prediction(policy, env, num_episodes, discount_factor = 1.0):
    policy = policy
    V = []
    return_sum = []
    return_count = []
    states_list  = []

    for episode in range(num_episodes):
        states = []
        G = []

        state = env.reset()
        while state[0] > 21 and state[0] < 12:
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
            state = next_state

        for state in range(len(states)):
            if states[state] in states_list:
                index = states_list.index(states_list[state])
                return_count[index] += 1
            else:
                states_list.append(states[state])
                return_count.append(1)
                return_sum.append(sum([x * (discount_factor ** i) for i, x in enumerate(G[state:])]))
    
    for state in range(len(states_list)):
        V.append([states_list[state], round(return_sum[state] / return_count[state], 2)])
    
    return np.array(V)

print(monte_carlo_prediction(20, env, 10000, discount_factor = 1.0))