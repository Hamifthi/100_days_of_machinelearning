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
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for episode in range(num_episodes):
        if (episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(episode + 1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        print(policy(state))
        action = np.random.choice(np.arange(len(policy(state))), p = policy(state))
        for t in itertools.count():
            next_state, reward, complete, _ = env.step(action)
            next_action = np.random.choice(np.arange(len(policy(next_state))), p = policy(next_state))
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = t
            Q[state][action] += alpha * (reward + (gamma * Q[next_state][next_action]) - Q[state][action])
            if complete:
                break
            state, action = next_state, next_action

    return Q, stats


Q, stats = sarsa(env, 200, gamma = 1.0, alpha = 0.5, epsilon = 0.1)
plotting.plot_episode_stats(stats)