import gym
import matplotlib
import numpy as np
from collections import defaultdict

env = gym.make('Blackjack-v0')

def make_epsilon_greedy_policy(Q, epsilon, nA):
    policy = np.zeros(2)
    def policy_function(observation):
        policy = [epsilon/env.action_space.n, epsilon/env.action_space.n]
        policy[np.argmax(Q[observation])] += 1 - epsilon
        return policy

    return policy_function

def mc_control_epsilon_greedy(env, num_episodes, gamma = 1.0, epsilon = 0.1):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for episode in range(num_episodes):
        state = env.reset()

        while True:
            states = []
            rewards = []
            G = 0

            states.append(state)
            action = np.argmax(policy(state))
            next_state, reward, complete, _ = env.step(action)
            rewards.append(reward)

            if complete == True:
                break
            state = next_state

        for step in states:
            

print(mc_control_epsilon_greedy(env, 10, gamma = 1.0, epsilon = 0.1))