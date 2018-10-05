import numpy as np
import matplotlib.pyplot as plt

def value_iteration_for_gamblers(probability_head, theta = 0.00001, gamma = 1.0):
    rewards = np.zeros(101)
    rewards[100] = 1
    V = np.zeros(101)

    def one_step_lookahead(state, V, rewards):
        q_values = np.zeros(101)
        for action in range(1, min(state, 100 - state) + 1):
            q_values[action] = (probability_head * (rewards[state + action] + gamma * V[state + action])) + ((1 - probability_head) * (rewards[state - action] + gamma * V[state - action]))
        return q_values

    while True:
        delta = 0
        for state in range(1, 100):
            q_values = one_step_lookahead(state, V, rewards)
            best_action_value = np.max(q_values)
            delta = max(delta, np.abs(best_action_value - V[state]))
            V[state] = best_action_value
        if theta > delta:
            break

    policy = np.zeros(100)
    for state in range(1, 100):
        q_values = one_step_lookahead(state, V, rewards)
        best_action = np.argmax(q_values)
        policy[state] = best_action

    return V, policy

v, policy = value_iteration_for_gamblers(0.25, theta = 0.00001, gamma = 1.0)

x = range(100)
# corresponding y axis values
y = v[:100]
 
# plotting the points 
plt.plot(x, y)
 
# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Value Estimates')
 
# giving a title to the graph
plt.title('Final Policy (action stake) vs State (Capital)')
 
# function to show the plot
plt.show()

# x axis values
x = range(100)
# corresponding y axis values
y = policy
 
# plotting the bars
plt.bar(x, y, align='center', alpha=0.5)
 
# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Final policy (stake)')
 
# giving a title to the graph
plt.title('Capital vs Final Policy')
 
# function to show the plot
plt.show()