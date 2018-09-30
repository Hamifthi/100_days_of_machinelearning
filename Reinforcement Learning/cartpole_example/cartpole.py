import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')
parameters = np.random.uniform(-1, 1, (4))

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(100):
        env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

def random_search_parameters():
    bestparameters = None
    bestreward = 0
    for _ in range(1000):
        parameters = np.random.uniform(-1, 1, (4))
        reward = run_episode(env, parameters)
        if reward > bestreward:
            bestreward = reward
            bestparameters = parameters
            if reward == 200:
                break         
    return bestparameters, bestreward

def hill_climbing_parameters():
    noise = 0.1
    parameters = np.random.uniform(-1, 1, (4))
    bestreward = 0
    for _ in range(1000):
        newparameters = parameters + np.random.uniform(-1, 1, (4)) * noise
        reward = 0
        reward = run_episode(env, newparameters)
        print("reward {} best {}".format(reward, bestreward))
        if reward > bestreward:
            bestreward = reward
            bestparameters = newparameters
            if reward == 200:
                break

hill_climbing_parameters()
try:
    del env
except ImportError:
    pass