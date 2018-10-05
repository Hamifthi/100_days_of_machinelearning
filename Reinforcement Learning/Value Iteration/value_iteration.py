import numpy as np
from gridworld import GridworldEnv

env = GridworldEnv()

def calculate_qvalues(env, v, gamma = 1.0):
    q_values = []
    for state in range(env.nS):
        q_state_action = np.zeros(env.nA)
        for action in range(env.nA):
            for probability, next_state, reward, done in env.P[state][action]:
                q_state_action[action] = probability * (reward + gamma * v[next_state])
        q_values.append(q_state_action)
    return np.array(q_values)

def value_iteration(env, gamma = 1.0, theta = 0.00001):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for state in range(env.nS):
            v_values = np.zeros(env.nA)
            for action in range(env.nA):
                for probability, next_state, reward, done in env.P[state][action]:
                    v_values[action] = probability * (reward + gamma * V[next_state])
            v = np.amax(v_values)
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
        if theta > delta:
            break
    q_values = calculate_qvalues(env, V, gamma = 1.0)
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for state in range(env.nS):
            for i in range(4):
                policy[state, i] = 1 if i == np.argmax(q_values[state]) else 0
    return {'v': np.array(V), 'policy': np.argmax(policy, axis = 1)}

print(value_iteration(env, gamma = 1.0, theta = 0.00001))