import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std= np.sqrt(2), bias_const= 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


'''Actor'''
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim= 256):
        super().__init__()
        # self.middle = nn.Sequential(
        #     nn.Linear(in_features= state_dim, out_features= hidden_dim),
        #     nn.Mish(),
        #     nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
        #     nn.Mish()
        # )
        # self.mean_layer = nn.Linear(in_features= hidden_dim, out_features= action_dim)
        # # 跟 SAC 一樣，輸出 log_std，後面會用 exp() 取出，以此來保持恆正
        # self.log_std_layer = nn.Linear(in_features= hidden_dim, out_features= action_dim)

        # 使用 Orthogonal init
        self.middle = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Mish(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Mish()
        )
        self.mean_layer = layer_init(nn.Linear(hidden_dim, action_dim), std= 0.01)
        self.log_std_layer = layer_init(nn.Linear(hidden_dim, action_dim), std= 0.01)
        
    
    # state : shape (batch_size, state_dim)
    # mean, std : shape (batch_size, action_dim)
    def forward(self, state):
        x = self.middle(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # 跟 SAC 一樣，限制 log_std 的範圍避免數值不穩。[-20, 2] 是公認的數值
        log_std = torch.clamp(log_std, min= -20, max= 2)
        std = torch.exp(log_std)
        return mean, std

    
    '''
        用在rollout 時用在產生 action (經過 tanh 的)
        回傳 new_log_prob 是因為此機率會在後面更新算修正項時被用到，因此會把這個值存在 rollout buffer 之中
    '''
    # state : shape (batch_size, state_dim)
    # action_tanh : shape (batch_size, action_dim)
    # new_log_prob : shape (batch_size, 1)
    def sample_action(self, state):
        mean, std = self.forward(state= state)
        # dist 為一個 Normal(mean, std) 的骰子，可以做重參抽樣、算 Entropy、算 log_prob
        # 因為 mean 跟 std 都是三維，因此這是一個三個獨立的高斯機率分布，做任何動作都會獨立的做這三個維度
        dist = torch.distributions.Normal(loc= mean, scale= std)
        # action is sampled by Reparameterizationd
        sample_action = dist.rsample()  # shape (batch_size, action_dim)
        # 雖然 PPO 沒有限制要用 tanh，但現在這種連續動作空間中加上 tanh 幾乎是標配，這是為了讓模型的輸出都盡量在合法的範圍內
        action_tanh = torch.tanh(sample_action)  # shape (batch_size, action_dim)
        # 計算 tanh 後的 log_prob : log_prob - tanh 修正項
        # log_prob : sample_action 的機率取 log
        original_log_prob = dist.log_prob(sample_action)  # shape (batch_size, action_dim)
        correction = torch.log(1 - action_tanh.pow(2) + 1e-6)
        # 相加是因為這邊要聯合動作機率的 log_prob，三個動作的分布互相獨立，因此聯合動作機率要相乘，又因為取 log，所以變相加
        new_log_prob = (original_log_prob - correction).sum(dim= 1, keepdim= True)  # shape (batch_size, 1)
        return action_tanh, new_log_prob

    
    '''用在更新時，我們要算出當前策略的 Log_prob & Entropy : 
        Log_prob : 公式上是說要 (current_prob(a_t|s_t) / old_prob(a_t|s_t))，這邊是在算 current_probs。取 log 的原因是為了數值穩定，可以用把除法變成減法，最後會再用 exp() 取出值
        Entropy : 這個是當前策略的 Entropy，這是為了避免 PPO 過早收斂進局部最佳。這在 2017 標準的 PPO 就有加在 loss function 了，只不過後面大家在講的時候都忽略
    '''
    # state : shape (batch_size, state_dim)
    # action : 這是舊策略做的 action，會用在取出該 action 在當前策略的 log_probs。shape (batch_size, action_dim)
    # current_log_prob : current_prob(a_t|s_t), shape (batch_size, 1)
    # entropy : 當前策略的 Entropy, shape (batch_size, 1)
    def evaluate(self, state, action):
        '''得出 curren_prob'''
        mean, std = self.forward(state= state)  # shape (batch_size, action_dim)
        # dist : current_prob
        dist = torch.distributions.Normal(loc= mean, scale= std)
        '''把當前 action 轉回 tanh 之前 by atanh'''
        # 先對 action clip，避免讓 action 是 -1 or 1，因為 atanh(1) = \infty、atanh(-1) = \infty
        action_clipped = torch.clamp(action, min= -0.999999, max= 0.999999)
        action_untanhed = 0.5 * (torch.log(1 + action_clipped) - torch.log(1 - action_clipped))
        '''現在有 current_prob(s_t) 也有原始 action (a_t) 了，可以取出 current_prob(a_t|s_t) 了'''
        current_log_prob = dist.log_prob(action_untanhed)  # shape (batch_size, action_dim)
        # 因為使用的是經過 tanh 的 action，所以求 log_prob 時一樣要經過 tanh 修正
        correction = torch.log(1 - action.pow(2) + 1e-6)
        current_log_prob = (current_log_prob - correction).sum(dim= 1, keepdim= True)  # shape (batch_size, 1)
        # 計算 Entropy 當作 loss 避免過早收斂到局部最佳
        entropy = dist.entropy().sum(dim= 1, keepdim= True)  # shape (batch_size, 1)
        return current_log_prob, entropy
        
        
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''Critic'''
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim= 256):
        super().__init__()
        # self.critic = nn.Sequential(
        #     nn.Linear(in_features= state_dim, out_features= hidden_dim),
        #     nn.Mish(),
        #     nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
        #     nn.Mish(),
        #     nn.Linear(in_features= hidden_dim, out_features= 1)
        # )
    
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Mish(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Mish(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0) # Value output std=1.0
        )

    # state : (batch_size, state_dim)
    # output : (batch_size, 1)
    def forward(self, state):
        return self.critic(state)




