import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

sys.path.append('E:/Hamed/Projects/Python/Machine Learning/100DaysOfMachineLearning/Reinforcement Learning/reinforcement-learning-master')
from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_function(observation):
        action_values = np.zeros(nA)
        action_values = list(map(lambda x: epsilon / nA, action_values))
        action_values[np.argmax(Q[observation])] += 1 - epsilon
        return action_values

    return policy_function

def q_learning(env, num_episodes, gamma = 1.0, alpha = 0.5, epsilon = 0.1):
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
        for t in itertools.count():
            action = np.random.choice(np.arange(len(policy(state))), p = policy(state))
            next_state, reward, complete, _ = env.step(action)
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = t
            Q[state][action] += alpha * (reward + (gamma * np.max(Q[next_state]) - Q[state][action]))
            if complete:
                break
            state = next_state

    return Q, stats

Q, stats = q_learning(env, 500, gamma = 1.0, alpha = 0.5, epsilon = 0.1)
plotting.plot_episode_stats(stats)