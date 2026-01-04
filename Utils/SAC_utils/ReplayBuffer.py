from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:


    def __init__(self, capacity, device):
        self.buffer = deque(maxlen= capacity)
        self.device = device

    
    def __len__(self):
        return len(self.buffer)
    

    def store_exp(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    
    def sample(self, batch_size):
        # zip 用來把垂直方向物件綁近一個 tuple 中 : 
        # a = [(1, 2, 3), (4, 5, 6)]
        # x, y, z = zip(*a)
        # print(x, y, z) : (1, 4) (2, 5) (3, 6)
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
        state = torch.from_numpy(np.array(state)).to(dtype= torch.float32, device= self.device)  # shape (batch_size, state_dim)
        action = torch.from_numpy(np.array(action)).to(dtype= torch.float32, device= self.device)  # shape (batch_size, action_dim)
        reward = torch.from_numpy(np.array(reward)).to(dtype= torch.float32, device= self.device).unsqueeze(dim= 1)  # shape (batch_size, 1)
        next_state = torch.from_numpy(np.array(next_state)).to(dtype= torch.float32, device= self.device)  # shape (batch_size, state_dim)
        return state, action, reward, next_state