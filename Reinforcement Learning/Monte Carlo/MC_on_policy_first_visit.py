import gym
import matplotlib
import numpy as np
import sys
import matplotlib.pyplot as plt
from plot import plot_value_function
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
    policy = defaultdict(int)
    policy_step = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for episode in range(num_episodes):
        if episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode, num_episodes), end="")
            sys.stdout.flush()
            
        state = env.reset()
        while state[0] > 21 and state[0] < 12:
            state = env.reset()

        while True:
            states = []
            actions = []
            rewards = []
            G = 0

            states.append(state)
            action = np.random.choice(np.arange(len(policy_step(state))), p = policy_step(state))
            actions.append(action)
            next_state, reward, complete, _ = env.step(action)
            rewards.append(reward)
            if complete == True:
                break
            state = next_state
            
        states_list = set(states)
        for step in range(len(states_list)):
            index = states.index(states[step])
            G = sum([x * (gamma ** i) for i, x in enumerate(rewards[index:])])
            returns_count[states[step]] += 1
            returns_sum[states[step]] += G
            Q[states[step]][actions[step]] = returns_sum[states[step]] / returns_count[states[step]]
            policy[states[step]] = np.argmax(Q[states[step]])
        policy_step = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    return Q, policy

def optimal_V(Q):
    V = defaultdict(float)
    for state, actions in Q.items():
        V[state] = np.max(actions)
    return V

Q, policy = mc_control_epsilon_greedy(env, 500000, gamma = 1.0, epsilon = 0.1)
V = optimal_V(Q)
plot_value_function(V, title="Optimal Value Function")
plt.show()