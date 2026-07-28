import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = False,
    ):
        feature = self.backbone(state)

        mean = self.mean_layer(feature)
        log_std = self.log_std_layer(feature)
        log_std = torch.clamp(
            log_std,
            min= self.log_std_min,
            max= self.log_std_max,
        )

        std = log_std.exp()
        distribution = Normal(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = distribution.rsample()

        if with_logprob:
            raw_log_prob = distribution.log_prob(raw_action)
            raw_log_prob = raw_log_prob.sum(dim=-1, keepdim=True)
        else:
            raw_log_prob = None

        return raw_action, mean, raw_log_prob
    


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