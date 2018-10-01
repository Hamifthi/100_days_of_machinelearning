import numpy as np
from gridworld import GridworldEnv
from policy_evaluation import policy_evaluation

env = GridworldEnv()

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_evaluation(random_policy, env)

def calculate_qvalues(env, v, gamma = 1.0):
    policy = np.zeros(env.nS)
    for state in range(env.nS):
        q_sa = np.zeros(env.nA)
        for action in range(env.nA):
            for probability, next_state, reward, done in env.P[state][action]:
                q_sa[action] = probability * (reward + gamma * v[next_state])
        policy[state] = np.argmax(q_sa)
    return policy

policy = calculate_qvalues(env, v)
print(policy)