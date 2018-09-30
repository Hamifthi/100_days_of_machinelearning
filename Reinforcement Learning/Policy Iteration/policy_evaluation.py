import numpy as np
from gridworld import GridworldEnv

env = GridworldEnv()

def policy_evaluation(policy, env, discount_factor = 1.0, theta = 0.00001):
    V = np.zeros(env.nS)
    delta = 0
    while True:
        for state in range(env.nS):
            v = 0
            for action, action_probability in enumerate(policy[state]):
                for probability, next_state, reward, done in env.P[state][action]:
                    v += action_probability * probability * (reward + discount_factor * V[next_state])
            if v != 0:
                if delta != 0:
                    delta = min(delta, np.abs(v - V[state]))
                else:
                    delta = np.abs(v - V[state])
            V[state] = v
        if theta > delta:
            break
    return np.array(V)

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_evaluation(random_policy, env)
print(v)