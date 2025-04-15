"""
This file will contain the Proximal Policy Agent (PPO).  
    
At each transistion we need to collect:
- state
- action
- advantage : This will need to be computed
- rewards
- value estimates

"""

import math
import os
import random
import gymnasium as gym
import ale_py
import numpy as np
import torch
import neural_ne
from torch.distributions.categorical import Categorical
import torch.nn.functional as func

# Hyperparamaters
# --------------------------------------------------------------------

class PPOAgent:
    
    def __init__ (self):
        
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_epsi = 0.2
        self.entropy_coef = 0.01
        self.actor = neural_ne.Actor(18)
        self.critic = neural_ne.Critic()
        
        self.information = {
                'state': [],
                'state_value_function': [],
                'action': [],
                'done': [],
                'log_prob_action': [],
                'reward': [],
                'cumulative_reward' : 0
            }
    
    def updateInformation (self, state, reward, done, trunc, info, action, action_prob):
        """Updates information from the action taken e.g., new states, rewards"""
        
        action = torch.tensor(action, dtype=torch.int64)
        
        self.information['state'].append(state)
        self.information['state_value_function'].append(self.get_state_value(state))
        self.information['done'].append(done)
        self.information['action'].append(action)
        self.information['log_prob_action'].append(action_prob)
        self.information['reward'].append(reward)
        self.information['cumulative_reward'] += reward
        
    def reset_information (self):
        """Resets information when the learning experience is over, ready for another one."""
        
        self.information = {
        'state': [],
        'state_value_function': [],
        'done': [],
        'action': [],
        'log_prob_action': [],
        'reward': [],
        'cumulative_reward' : 0
    }
    
    def compute_advantages (self, timestamp, state_value_function):
        """Computes advantages the long way - probably won't be used now"""
        rewards_after_state = self.information['reward'][timestamp:]
        return (sum([rewards_after_state[x] ^ (x+1) for x in range(len(rewards_after_state))]) - state_value_function)
    
    # This needs some more work
    def compute_gen_advantage_estimation (self):
        """Returns the general advantage estimation"""
        
        gae = 0
        returns = []
        rewards = self.information['reward']
        state_value = self.information['state_value_function']
        done = self.information['done']
        mask = [1 if x is False else 0 for x in done]
        
        for i in range(len(rewards)-2,-1,-1):
            delta = rewards[i] + (self.gamma * mask[i] * state_value[i+1]) - state_value[i]
            gae = delta + (self.gamma * mask[i] * self.lam * gae)
            returns.insert(0, gae + state_value[i])
        
        adv = torch.tensor(np.array(returns) - state_value[:-1], dtype=torch.float32)
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-10)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        return returns, adv
    
    
    def clipped_surrogate_loss(self, advantage, old_probability, new_probability):
        """Calculates the clipped value"""
        ratio = new_probability/old_probability
        
        unclipped_loss = ratio * advantage
        clipped_loss = torch.clamp(ratio, 1-self.clip_epsi, 1+self.clip_epsi) * advantage
        
        return torch.minimum(unclipped_loss, clipped_loss).mean()
    
    def get_action (self, state, evaluate=False):
        """Gets action from actor NN
        
        Inputs: State and Evaluate (bool)
        Outputs: Selected action, probability of action and entropy
        
        """
        with torch.no_grad():
            # Returns the output of the actor neural net
            logits = self.actor.forward(state)
            # Distribution object
            dist = Categorical(logits=logits)
        
            if evaluate:
                action = torch.argmax(logits, dim=1)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item()
        
        
    def get_state_value (self, state):
        with torch.no_grad():
            output = self.critic.forward(state)
            value = torch.squeeze(output).item()
            return value
    
    def calculate_losses (self, surrogate_loss, entropy, returns, value_predictions):
        entropy_bonus = self.entropy_coef * entropy
        policy_loss = (-surrogate_loss + entropy_bonus).sum()
        value_loss = func.smooth_l1_loss(returns, value_predictions).sum()
        return policy_loss, value_loss
    
    @staticmethod
    def state_manipulation (to_grayscale, state):
        
        if isinstance(state, tuple):
            state = state[0]
        
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1) / 255.0
        state = to_grayscale(state)
        state = state.unsqueeze(0)
        return state
    
    def learn (self, batch_size = 64, epochs = 10):
        # Get all information from the agent
        states = torch.stack(self.information['state'])
        states = states.squeeze(1)
        actions = torch.stack(self.information['action'])
        old_action_prob = torch.tensor(self.information['log_prob_action'], dtype = torch.float32)
        rewards = torch.tensor(self.information['reward'], dtype = torch.float32)
        state_value = torch.tensor(self.information['state_value_function'], dtype=torch.float32)
        done = self.information['done']
        
        # Computes returns and advantages ready for NN training
        returns, advantages = self.compute_gen_advantage_estimation()
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        dataset_size = len(done)
        
        for _ in range(epochs):
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, batch_size):
                
                end = start + batch_size
                if end > dataset_size:
                    end = dataset_size
                
                batch_idx = indices[start:end].to(torch.long)
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_action_prob = old_action_prob[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_state_value = state_value[batch_idx]
        
                print(1)
 
                logits = self.actor.forward(batch_states)
                dist = Categorical(logits=logits)
                new_action_prob = dist.log_prob(batch_actions)
        
                entropy_loss = torch.mean(dist.entropy())
                
                surrogate_loss = self.clipped_surrogate_loss(batch_advantages, batch_old_action_prob, new_action_prob)
                
                policy_loss, value_loss = self.calculate_losses(surrogate_loss, entropy_loss, batch_returns, batch_state_value)
                
                total_loss = policy_loss + 0.5 * value_loss
                print(2)
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                
                total_loss.backward()
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                print(3)
        self.reset_information()
    
    def access_cumulative_reward (self):
        return self.information['cumulative_reward']

    def access_information(self):
        return self.information