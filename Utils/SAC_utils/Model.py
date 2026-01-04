import torch
import torch.nn as nn
import torch.nn.functional as F


'''SAC Actor'''
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim= 256):
        super().__init__()
        self.middle_layer = nn.Sequential(
            nn.Linear(in_features= state_dim, out_features= hidden_dim),
            nn.Mish(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
            nn.Mish(),
        )
        self.mean_layer = nn.Linear(in_features= hidden_dim, out_features= action_dim)
        self.std_layer = nn.Linear(in_features= hidden_dim, out_features= action_dim)
    

    def forward(self, state):
        mid = self.middle_layer(state)
        mean = self.mean_layer(mid)
        # PPO、SAC 跟大多處理連續動作空間的演算法都是輸出 log_std，這是因為 std 必須保證是恆正，使用 log 再用 e 取出能保證恆正
        log_std = self.std_layer(mid)
        # classic SAC clamping tricks
        # [-20, 2] 是有深刻的數學考量的 : 因為 exp 是增長快速的函數，若不控制其 x，則會導致數字超大，畢竟神經網路的輸出是 [-infty, -infty]
        # 此外，若 x 超小，會導致 exp 趨近於 0，就容易發生 log(0) 和喪失探索能力的情況
        # 因此，為了數值穩定不崩潰，將 log_std 訂於 [-20, 2]
        # e^2 = 7.389, e^(-20) = 2x10^(-9)
        # 這組 [-20, 2] 的參數設定在 MuJoCo、Atari 等表現最穩定。
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std
        
    
    # state : (batch_size, state_dim)
    # output : actions, new_log_probs : (batch_size, action_dim), (batch_size, 1)
    def sample_action(self, state):
        # mean, std : (batch_size, action_dim), (batch_size, action_dim)
        mean, std = self.forward(state= state)
        # dists : action_dim 個高斯分布
        # loc : mean, scale : std
        # 可以對 dist 做重參抽樣 (rsample())、算 Entropy、算 log_prob
        dists = torch.distributions.Normal(loc= mean, scale= std)
        # 對 dists 中每一個機率分布中做重參抽樣
        # logits : (batch_size, action_dim)
        logits = dists.rsample()
        # 限縮 logits 於 [-1, 1] by tanh (Classic SAC)
        actions = torch.tanh(logits)  # shape (batch_size, action_dim)
        # 計算 tanh 後的機率分布的 Entropy (即原始機率分布的 Entropy - 使用 Tanh 的修正項)
        original_log_probs = dists.log_prob(logits)  # shape (batch_size, action_dim)
        correction = torch.log(1 - actions.pow(2) + 1e-6)  # shape (batch_size, action_dim)
        # new_log_probs 為該 batch 中各資料的 -Entropy
        # 後面那個 .sum 是把三個維度的 Entropy 相加 (SAC 假設不同維度的分布是獨立的，因此聯合機率是用乘的，因為取 log 所以變加的)
        # 若 keepdim= False，則 shape (batch_size)
        # 若 keepdim= True，則 shape (batch_size, 1)
        new_log_probs = (original_log_probs - correction).sum(dim= 1, keepdim= True)
        return actions, new_log_probs



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''SAC Critic'''
class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim= 256):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(in_features= state_dim + action_dim, out_features= hidden_dim),
            nn.Mish(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
            nn.Mish(),
            nn.Linear(in_features= hidden_dim, out_features= 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(in_features= state_dim + action_dim, out_features= hidden_dim),
            nn.Mish(),
            nn.Linear(in_features= hidden_dim, out_features= hidden_dim),
            nn.Mish(),
            nn.Linear(in_features= hidden_dim, out_features= 1)
        )
    

    # state : shape (batch_size, state_dim)
    # action : shape (batch_size, action_dim)
    # q1, q2 : shape (batch_size, 1)
    def forward(self, state, action):
        # shape (batch_size, state_dim + action_dim)
        input = torch.cat([state, action], dim= 1)
        q1, q2 = self.Q1(input), self.Q2(input)
        return q1, q2

    
    # state : shape (batch_size, state_dim)
    # action : shape (batch_size, action_dim)
    # output : shape (batch_size, 1)
    def q_min(self, state, action):
        return torch.min(*self.forward(state, action))


