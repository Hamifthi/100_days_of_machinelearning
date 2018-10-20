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
        return action_values
    return policy_function

def create_greedy_policy(Q):
    def policy_function(observation):
        return np.argmax(Q[observation])
    return policy_function

def mc_control_importance_sampling(env, num_episodes, behavior_policy, gamma = 1.0):
    C = defaultdict(lambda: np.zeros(env.action_space.n))
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
            action = np.random.choice(len(behavior_policy(state)), p = behavior_policy(state))
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
            policy[state] = target_policy(state)
            if action !=  target_policy(state):
                break
        W = W * 1. / behavior_policy(state)
        target_policy = create_greedy_policy(Q)

    return Q, policy

def optimal_V(Q):
    V = defaultdict(float)
    for state, actions in Q.items():
        V[state] = np.max(actions)
    return V

random_policy = create_random_policy(2)
Q, policy = mc_control_importance_sampling(env, 500000, random_policy, gamma = 1.0)
V = optimal_V(Q)
plot_value_function(V, title="Optimal Value Function")