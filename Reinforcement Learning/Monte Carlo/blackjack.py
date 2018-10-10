import gym
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import axes3d, Axes3D

env = gym.make('Blackjack-v0')

def monte_carlo_prediction(policy, env, num_episodes, discount_factor = 1.0):
    policy = policy
    V = defaultdict(float)
    return_sum = defaultdict(float)
    return_count = defaultdict(float)

    for episode in range(num_episodes):
        states = []
        G = []

        state = env.reset()
        while state[0] > 21 and state[0] < 12:
            state = env.reset()

        while True:
            states.append(state)
            if state[0] == policy:
                next_state, reward, complete, _ = env.step(0)
            else:
                next_state, reward, complete, _ = env.step(1)
            G.append(reward)
            if complete == True:
                break
            state = next_state

        states_list = set(states)
        for state in states_list:
            index = states.index(state)
            return_sum[state] += sum([x * (discount_factor ** i) for i, x in enumerate(G[index:])])
            return_count[state] += 1.0
            V[state] = return_sum[state] / return_count[state]

    return V

# using this repository for ploting address https://github.com/dennybritz/reinforcement-learning
def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = Axes3D(fig)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

V_10k = monte_carlo_prediction(20, env, num_episodes = 10000)
plot_value_function(V_10k, title="10,000 Steps")
plt.show()

V_500k = monte_carlo_prediction(20, env, num_episodes = 500000)
plot_value_function(V_500k, title="500,000 Steps")
plt.show()