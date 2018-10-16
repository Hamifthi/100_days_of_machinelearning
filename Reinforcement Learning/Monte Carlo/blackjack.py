import gym
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from plot import plot_value_function
from collections import defaultdict
from mpl_toolkits.mplot3d import axes3d, Axes3D

env = gym.make('Blackjack-v0')

def monte_carlo_prediction(policy, env, num_episodes, discount_factor = 1.0):
    policy = policy
    V = defaultdict(float)
    return_sum = defaultdict(float)
    return_count = defaultdict(float)

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

        states_list = set(states)
        for state in states_list:
            index = states.index(state)
            return_sum[state] += sum([x * (discount_factor ** i) for i, x in enumerate(G[index:])])
            return_count[state] += 1.0
            V[state] = return_sum[state] / return_count[state]

    return V

V_10k = monte_carlo_prediction(20, env, num_episodes = 10000)
plot_value_function(V_10k, title="10,000 Steps")
plt.show()

V_500k = monte_carlo_prediction(20, env, num_episodes = 500000)
plot_value_function(V_500k, title="500,000 Steps")
plt.show()