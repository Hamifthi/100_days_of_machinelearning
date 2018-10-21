import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

sys.path.append('E:/Hamed/Projects/Python/Machine Learning/100DaysOfMachineLearning/Reinforcement Learning/reinforcement-learning-master')
from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = WindyGridworldEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_function(observation):
        action_values = np.zeros(nA)
        action_values = list(map(lambda x: epsilon / nA, action_values))
        action_values[np.argmax(Q[observation])] += 1 - epsilon
        return action_values

    return policy_function

def sarsa(env, num_episodes, gamma = 1.0, alpha = 0.5, epsilon = 0.1):
    for episode in range(num_episodes):
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        Q[37][0] = 0
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        if (episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(episode + 1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        action = np.random.choice(np.arange(len(policy(state))), p = policy(state))
        while True:
            next_state, reward, complete, _ = env.step(action)
            next_action = np.random.choice(np.arange(len(policy(next_state))), p = policy(next_state))
            Q[state][action] += alpha * (reward + (gamma * Q[next_state][next_action]) - Q[state][action])
            state, action = next_state, next_action
            if state == 37:
                break
        
    return Q


print(sarsa(env, 100, gamma = 1.0, alpha = 0.5, epsilon = 0.1))