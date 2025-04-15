import os
import random
import gymnasium as gym
import ale_py
import numpy as np
import ppo
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T


env = gym.make("ALE/Boxing-v5")
agent = ppo.PPOAgent()
to_grayscale = T.Grayscale(num_output_channels=1)

def gather_data():
    """Gathers data for a full episode"""
    cumulative_reward = 0
    while cumulative_reward < 80:
        done = False
        state = env.reset()

        while not done:
            
            # Turns it into a tensor for neural net input
            
            state = agent.state_manipulation(to_grayscale, state)
                
            # Retrieves the action to be taken
            action, prob = agent.get_action(state)
            action = torch.tensor(action, dtype=torch.int64)
            
            
            # Executes function and retrieves information
            new_state, reward, done, trunc, info = env.step(action)
            
            # Updates the information for that timestamp into the information dictionary
            agent.updateInformation(state, reward, done, trunc, info, action, prob)
            
            state = new_state
        cumulative_reward = agent.access_cumulative_reward()
        agent.learn()
            

def output_to_excel(info: dict[list]):
    info_to_export = {k: v for k, v in info.items() if isinstance(v, list)}
    
    df = pd.DataFrame(info_to_export)
    df.to_excel("output.xlsx", index=False)

gather_data()