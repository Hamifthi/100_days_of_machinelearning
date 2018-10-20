import gym
import matplotlib
import numpy as np
import sys
import matplotlib.pyplot as plt
from plot import plot_value_function
from collections import defaultdict

env = gym.make('Blackjack-v0')

def create_random_policy(nA):
    action_values = np.ones(nA) / nA
    def policy_function(observation):
        return np.random.choice(len(action_values), p = action_values)
    return policy_function

def create_greedy_policy(Q):
    def policy_function(observation):
        return np.argmax(Q[observation])
    return policy_function

def mc_control_importance_sampling(env, num_episodes, behavior_policy, gamma = 1.0):
    C = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    target_policy = create_greedy_policy(Q)
    policy = defaultdict(int)

    for episode in range(num_episodes):
        if episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        episode = []
        G = 0
        W = 1

        while True:
            action = behavior_policy(state)
            next_state, reward, complete, _ = env.step(action)
            episode.append((state, action, reward))
            if complete:
                break
            state = next_state
        
        for step in range(len(episode))[::-1]:
            state, action, reward = episode[step]
            G = G * gamma + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action !=  target_policy(state):
                break
            W = W * 1. / behavior_policy(state)[action]
            policy(states[step]) = target_policy(states[step])
        W = W / behavior_policy



random_policy = create_random_policy(2)
print(mc_control_importance_sampling(env, 10, random_policy, gamma = 1.0))