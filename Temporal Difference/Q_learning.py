#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 00:23:08 2025

@author: HS
"""
# Q-learning implimentation
import time
import gymnasium as gym
import numpy as np

# modes: ['human', 'ansi', 'rgb_array']
# is_slippery decide if the environment is stochastic or deterministic
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False).unwrapped

class TemporalDifference():
    def __init__(self, num_iterations, epsilon, alpha, discount_factor):
        # Get the number of actions and states to create 2D arrays
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        
        # Theoricly we consider an infinit loop while praticly we loop for a fixed iteration number
        # Or until the difference between old value-function and current value-function is less than a thresholder theta
        self.num_iterations = num_iterations
        
        # algorithm parameters
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount_factor = discount_factor
       
    def SARSA(self):
        # Initialize Q(s,a) arbitrarly or zeros
        # Q = np.random.uniform(0, 0.1, size =(self.num_states, self.num_actions)) # require so many iterations
        Q = np.zeros((self.num_states, self.num_actions))
        t_begin = time.time()
        
        for _ in range(self.num_iterations):
            current_state = env.reset()[0]
            done = False
            current_action = self.epsilonGreedy(current_state, Q)
        
            while not done:
                next_state, reward, term, trunk, _ = env.step(current_action)
                next_action = self.epsilonGreedy(next_state, Q)
                
                done = term or trunk
                Q[current_state, current_action] += self.alpha *\
                    (reward + (self.discount_factor * np.max(Q[next_state])) - Q[current_state, current_action])
                
                current_action = next_action
                current_state = next_state
        print(f"time: {time.time() - t_begin:0.3f}s")
        policy = np.argmax(Q, axis=1)
        return policy
    
    def epsilonGreedy(self, state, Q):
        if self.epsilon > np.random.rand():
            return np.random.randint(0, self.num_actions)
        return np.argmax(Q[state])
    
        
num_iterations = 80_000
# time without policy improvement = 0.549ms
epsilon = 0.5
alpha = 0.79
discount_factor = 0.9
np.random.seed(1)

# [1 0 0 0 1 1 0 1 2 1 0 1 0 2 2 0]
# [2 2 1 2 3 0 1 0 0 2 1 0 0 2 2 0]
# [1 0 0 0 1 0 1 0 2 2 1 0 0 2 2 0]

TD = TemporalDifference(num_iterations, epsilon, alpha, discount_factor)

policy = TD.SARSA()
print(policy)

env.close()

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False).unwrapped
env = gym.wrappers.TimeLimit(env, max_episode_steps=20)
state = env.reset()[0]


done = False
total_reward = 0 
while not done:
    action = policy[state]
    state_next, reward, term, trunk, _ = env.step(action)
    done = term or trunk
    state = state_next
    env.render()
    total_reward += reward

env.close()
