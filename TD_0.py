#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 23:58:22 2025

@author: home
"""

# TD(0): one-step look ahead
import time
import gymnasium as gym
import numpy as np

# modes: ['human', 'ansi', 'rgb_array']
# is_slippery decide if the environment is stochastic or deterministic
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False).unwrapped

class TemporalDifference(): # TD(0)
    def __init__(self, num_iterations, alpha, discount_factor):
        # Get the number of actions and states to create 2D arrays
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        
        # Theoricly we consider an infinit loop while praticly we loop for a fixed iteration number
        # Or until the difference between old value-function and current value-function is less than a thresholder theta
        self.num_iterations = num_iterations
        
        self.discount_factor = discount_factor
        # To insure exploration avoiding local optimum
        self.alpha = alpha
        
    
    def OneStep(self, policy):
        # Initialize Q(s,a) by 0
        V = np.random.randint(0, self.num_actions, size = self.num_states)
        td_error = 0
        
        # Initialize the policy arbitrarily
        policy = np.random.randint(0, self.num_actions, self.num_states)
        
        t_begin = time.time()
        
        for _ in range(self.num_iterations):
            # episode = self.EpisodeGenerator(policy)
            current_state = env.reset()[0]
            done = False
            while not done:    
                action = policy[current_state]
                next_state, reward, term, trunk, _ = env.step(action)
                done = term or trunk
                V[current_state] = V[current_state] + self.alpha * (reward + self.discount_factor * V[next_state] - V[current_state])
                current_state = next_state
                
                
        print(f"time: {time.time() - t_begin:0.3f}s")
            
        return V
    
        
        
        
num_iterations = 800
# time without policy improvement = 0.549ms
alpha = 0.1
discount_factor = 0.5
np.random.seed(1)

# Example of policies

# Optimal policy
policy = [1, 0, 0, 0, 1, 1, 0, 1, 2, 1, 0, 1, 0, 2, 2, 0]

TD_0 = TemporalDifference(num_iterations, alpha, discount_factor)
state_value_function = TD_0.OneStep(policy)
print(state_value_function)

env.close()

