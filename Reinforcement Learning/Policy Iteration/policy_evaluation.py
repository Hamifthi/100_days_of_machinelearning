import numpy as np
import time
from gridworld import GridworldEnv

env = GridworldEnv()

def policy_evaluation(policy, env, gamma = 1.0, theta = 0.00001):
    V = np.zeros(env.nS)
    v_old = np.zeros(env.nS)
    while True:
        for state in range(env.nS):
            v = 0
            for action, action_probability in enumerate(policy[state]):
                for probability, next_state, reward, done in env.P[state][action]:
                    v += action_probability * probability * (reward + gamma * V[next_state])
            v_old[state] = V[state]
            V[state] = v
        if theta > np.amax(np.abs(V - v)):
            break
    return np.array(V)

if __name__ == '__main__':
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v_values = policy_evaluation(random_policy, env)
    print(v_values)