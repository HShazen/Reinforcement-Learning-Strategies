#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 19:00:31 2025

@author: HS
"""
# This program impliment a basic Monte Carlo algorithm and return a deterministic policy π*
import time
import gymnasium as gym
import numpy as np

# modes: ['human', 'ansi', 'rgb_array']
# is_slippery decide if the environment is stochastic or deterministic
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False).unwrapped

class MonteCarlo():
    def __init__(self, num_iterations, epsilon, discount_factor):
        # Get the number of actions and states to create 2D arrays
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        
        # Theoricly we consider an infinit loop while praticly we loop for a fixed iteration number
        # Or until the difference between old value-function and current value-function is less than a thresholder theta
        self.num_iterations = num_iterations
        
        # To insure exploration avoiding local optimum
        self.epsilon = epsilon
        
        # The discount factor (gamma) determines how much future rewards are valued compared to immediate rewards
        self.discount_factor = discount_factor
        
    
    def FirstVisit(self):
        # Initialize Q(s,a) by 0
        Q = np.zeros((self.num_states, self.num_actions))
        
        returns = np.zeros((self.num_states, self.num_actions))
        num_occurences = np.zeros((self.num_states, self.num_actions))
        
        # Initialize the policy arbitrarily
        policy = np.random.randint(0, self.num_actions, self.num_states)
        
        t_begin = time.time()
        
        for _ in range(self.num_iterations):
            episode = self.EpisodeGenerator(policy)
            G = 0
            visits = []
            for step in episode[::-1]:
                # step is a tuple (state, action, reward)
                state, action, reward = step
                if (state, action) in visits:
                    visits.append((state, action))
                G = self.discount_factor * G + reward
                returns[state, action] += G
                num_occurences += 1
                Q[state, action] = returns[state,action] / num_occurences[state,action]
                optimal_action = np.argmax(Q[state])
                # for now we consider deterministic policy
                policy[state] = optimal_action
                
        print(f"time: {time.time() - t_begin:0.3f}s")
            
        return policy 
    
    def EpisodeGenerator(self, policy):
        episode = []
        current_state = env.reset()[0]
        done = False
        while not done:
            action = policy[current_state]
            if self.epsilon > np.random.rand():
                action = np.random.randint(0, self.num_actions)
            next_state, reward, term, trunk, _ = env.step(action)
            done = term or trunk
            episode.append((current_state, action, reward))
            current_state = next_state
            
        return episode
        
        
        
num_iterations = 8000
# time without policy improvement = 0.549ms
epsilon = 0.1
discount_factor = 0.5
np.random.seed(1)


MC = MonteCarlo(num_iterations, epsilon, discount_factor)
policy = MC.FirstVisit()
print(policy)

env.close()

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False, max_episode_steps=30).unwrapped
state = env.reset()[0]


done = False
total_reward = 0 
i = 0
while not done:
    action = policy[state]
    state_next, reward, term, trunk, _ = env.step(action)
    done = term or trunk
    state = state_next
    env.render()
    total_reward += reward
    i += 1
    if i == 20:
        break

