import numpy as np
from gridworld import GridworldEnv
from policy_evaluation import policy_evaluation

env = GridworldEnv()

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_evaluation(random_policy, env)

def calculate_qvalues(env, v, gamma = 1.0):
    q_values = []
    for state in range(env.nS):
        q_state_action = np.zeros(env.nA)
        for action in range(env.nA):
            for probability, next_state, reward, done in env.P[state][action]:
                q_state_action[action] = probability * (reward + gamma * v[next_state])
        q_values.append(q_state_action)
    return np.array(q_values)

def policy_improvement(env, policy_eval_fn = policy_evaluation, gamma = 1.0):
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        v_values = policy_evaluation(policy, env)
        q_values = calculate_qvalues(env, v_values, gamma = 1.0)
        old_policy = policy

        for state in range(env.nS):
            for i in range(4):
                policy[state, i] = 1 if i == np.argmax(q_values[state]) else 0

        if (old_policy == policy).all():
            break

    return policy

policy = policy_improvement(env, policy_eval_fn = policy_evaluation, gamma = 1.0)
V = policy_evaluation(policy, env)
print(V)