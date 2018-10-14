import gym
import matplotlib
import numpy as np
from collections import defaultdict

env = gym.make('Blackjack-v0')

def make_epsilon_greedy_policy(Q, epsilon, nA):
    policy = defaultdict(float)
    def policy_function(observation):
        index = np.argmax(Q[observation])
        policy[observation] = [epsilon/env.action_space.n, epsilon/env.action_space.n]
        policy[observation][index] += 1 - epsilon
        return policy

    return policy_function

Q = defaultdict(lambda: np.zeros(env.action_space.n))
print(make_epsilon_greedy_policy(Q, 0.5, env.action_space.n)(env.observation_space.sample()))