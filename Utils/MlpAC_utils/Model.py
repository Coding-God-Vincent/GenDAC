import torch
import torch.nn as nn
from torch.distributions import Normal

'''Actor (by MLP)'''
'''這邊加入 Entropy 的原因是因為 GenDAC 多步就有 Entropy 的意味，這邊若沒有的話很快會陷入局部最優'''
class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, using_tanh= False):
        super().__init__()
        self.middle = nn.Sequential(
            nn.Linear(in_features= state_dim, out_features= 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish()
        )
        self.mean_layer = nn.Linear(in_features= 256, out_features= action_dim)
        self.log_std_layer = nn.Linear(in_features= 256, out_features= action_dim)
        # 只要是輸出 Std 都會用這招
        # 為了 Std 恆正，會轉成 log，使用時再用 exp 取出 (exp 取出的值恆正)
        # 為了避免數值爆炸或消失，將 log std 限縮在 -2 ~ 20 之間 (Magic No)
        self.LOG_STD_MAX = 20
        self.LOG_STD_MIN = -2
        self.using_tanh = using_tanh
        
    
    # state : shape (batch_size, state_dim)
    # deterministic : bool, True 時為取 Mean，用於 evaluation、False 時用於 Training
    # with_logprob : bool, True 時會計算當前機率分布的 entropy，用於 Entropy 的計算
    # return : 
    # action : shape (batch_size, action_dim)
    # log_prob : shape (batch_size, 1)
    def forward(self, state, deterministic= False, with_logprob= True):
        extracted_state = self.middle(state)
        mean = self.mean_layer(extracted_state)  # (batch_size, action_dim)
        log_std = self.log_std_layer(extracted_state)  # (batch_size, action_dim)
        # clipped log_std for stablization
        log_std = torch.clamp(log_std, min= self.LOG_STD_MIN, max= self.LOG_STD_MAX)
        # use exponential to extract std
        std = torch.exp(log_std)  # shape (batch_size, action_dim)
        dist = Normal(loc= mean, scale= std)  # 3 independent Gaussian Dist.
        # (Determinisitic = True) evaluation : action = mean
        # (Deterministic = False) training : action is sampled by reparameterization
        if deterministic: logits = mean
        else: logits = dist.rsample()
        
        if self.using_tanh:
            # we use clamp to contraint the values in [-1, 1] in GenDAC
            # so we use tanh here to do the same things
            action = torch.tanh(logits)
            # (with_log_prob = True) compute log_prob in order to evaluate Entropy
            # (with_log_prob = False) return None
            if with_logprob:
                # because the action is tanh(u) so the corresponding log_prob must be corrected 
                log_prob = dist.log_prob(logits) - torch.log(1 - action.pow(2) + 1e-6)
                # 這三維是三個獨立的機率分布，故其聯合 Entropy 可以直接用乘的，而這邊是 log 所以用加的
                log_prob = log_prob.sum(dim= -1, keepdim= True)
            else: log_prob = None
        else:
            action = logits
            if with_logprob:
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            else:
                log_prob = None
            return action, log_prob
        return logits, log_prob


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''Double Critic (Same as GenDAC)
* 2 inputs:
    1. state (s_t) : shape (batch_size, state_dim)
    2. action (a_t) : shape (batch_size, action_dim)
* 2 kinds of outputs:
    1. outputs of 2 Q_networks : shape ((batch_size, 1), (batch_size, 1))
    2. min of 2 Q_networks : shape (batch_size, 1)
'''

class DoubleCritic(nn.Module):
    def __init__(
        self, 
        state_dim,
        action_dim,
        hidden_dim= 256,
        activation= 'mish'
    ):
        super().__init__()
        # activation function
        act = nn.Mish if activation == 'mish' else nn.ReLU
        # State Embedding Layer
        self.state_mlp = nn.Sequential(
            nn.Linear(in_features= state_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
            act()
        )
        # Q1
        self.Q_network1 = nn.Sequential(
            nn.Linear(in_features= hidden_dim + action_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= 1)
        )
        # Q2
        self.Q_network2 = nn.Sequential(
            nn.Linear(in_features= hidden_dim + action_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
            act(),
            nn.Linear(in_features= hidden_dim, out_features= 1)
        )


    # return : shape (batch_size, action_dim), (batch_size, action_dim)
    def forward(self, state, action):
        embedding_state = self.state_mlp(state)
        total_input = torch.cat([embedding_state, action], dim= 1)  # shape (batch_size, hidden_dim + action_dim)
        return self.Q_network1(total_input), self.Q_network2(total_input)
    
    
    # return : shape (batch_size, 1)
    def q_min(self, state, action):
        return torch.min(*self.forward(state, action))

