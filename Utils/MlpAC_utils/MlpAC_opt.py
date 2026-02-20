import torch
import torch.nn.functional as F
import numpy as np
import copy
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou import BasePolicy

class MlpAC_opt(BasePolicy):
    def __init__(
        self, 
        actor, 
        actor_optim,
        critic,
        critic_optim,
        device, 
        state_dim,
        action_dim,
        gamma= 0.99,
        tau= 0.005,
        alpha_lr= 3e-4,
        n_step = 3
    ):
        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_target = copy.deepcopy(actor)
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_target = copy.deepcopy(critic)
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        
        # Auto-Tuning Alpha (2018 SAC) : proportion of the entropy
        # alpha must be positive so we use log-exp trick here as well
        # target_entropy is set to -dim(action) ususally (experience)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad= True, device= device)
        self.alpha_optim = torch.optim.Adam(params= [self.log_alpha], lr= alpha_lr)
        self.alpha = self.log_alpha.exp().item()  # int
        
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 batch 的資料，根據 obs/obs_ 傳入 actor/target_actor 後得到該 batch 各資料的 act，並把結果放到一個新的 batch 後回傳該新 batch
    # state : str  # indicate to use obs or obs_next
    # model : str  # indicate to use actor or target_actor
    # return : 計算的結果，用一個 Batch 裝起來
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    '''essential in BasePolicy'''
    '''this function is like forward() in nn.Module'''
    def forward(
        self,
        batch : Batch,
        state : str = 'obs',
        model : str = 'actor'
    ):
        # shape (batch_size, state_dim)
        state = to_torch(batch[state], dtype= torch.tensor32, device= self.device)
        model_ = self.actor if model == 'actor' else self.actor_target
        # return action, log_prob
        action, _ = model_(state)
        return Batch(act= action)
    
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入當前 Batch 的資料，算出各資料的 Critic^k_loss 中最後的 min{ Q_target^k(s(t+n), a_target(t+n) }，即 TD Target 的 Q 部分
    # 不用考慮 n_step，這邊只是在設定如何透過 target_actor 得出 min{Q_target}
    # 這邊傳入的 batch = buffer[indices]，batch.obs_next 已經是 s(t+n) (process_fn 所呼叫的 compute_nstep_return 會算前面的)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def td_target_q(
        self,
        buffer : ReplayBuffer, 
        indices : np.ndarray  # shape (batch_size)
    ):
        pass
    
    
    # sample_size : int, batch_size
    # buffer : ReplayBuffer of tianshou
    '''essential in BasePolicy'''
    def update(self, sample_size, buffer):
        # batch : tianshou datatype : batch
        batch = buffer.sample(sample_size) 
        # shape (batch_size, state_dim)
        state = torch.tensor(batch.obs, dtype= torch.float32, device= self.device)
        # shape (batch_size, action_dim)
        action = torch.tensor(batch.act, dtype= torch.float, device= self.device)
        # shape (batch_size, state_dim)
        next_state = torch.tensor(batch.obs_next, dtype= torch.float32, device= self.device)
        # shape (batch_size, 1)
        reward = torch.tensor(batch.rew, dtype= torch.float32, device= self.device).view(-1, 1)
        
        # current alpha
        alpha = self.log_alpha.exp()
        
        '''update Critic'''
        with torch.no_grad():
            # next_action : shape (batch_size, action_dim)
            # next_log_prob : shape (batch_size, 1)
            next_action, next_log_prob = self.actor_target(next_state)
            target_Q = self.critic_target.q_min(state= next_state, action= next_action) - alpha.detach() * next_log_prob
            target_Q = reward + self.gamma * target_Q
        
        current_Q1, current_Q2 = self.critic(state= state, action= action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        '''Update Actor'''
        # current_action : shape (batch_size, action_dim)
        # current_log_prob : shape (batch_size, 1)
        current_action, current_log_prob = self.actor(state= state)
        # shape (batch_size, 1)
        currentQ = self.critic.q_min(state= state, action= current_action)
        actor_loss = (alpha.detach() * current_log_prob - currentQ).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        '''Update Alpha'''
        alpha_loss = -(self.log_alpha * (current_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()  # update currently used alpha
        
        '''Soft update target networks'''
        # 把每一個 critic 參數跟對應的 target_critic 中的參數組成一個 tuple (by zip) 後取出進行 Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            # .data 取出純數值，因為我們不要梯度
            # copy_ : 代表 in-place，不另外創建一個 tensor
            # copy_(...) 中 ... 為 target critic update 的目標
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            # .data 取出純數值，因為我們不要梯度
            # copy_ : 代表 in-place，不另外創建一個 tensor
            # copy_(...) 中 ... 為 target critic update 的目標
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # return values : 4 int
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'policy_entropy': -current_log_prob.mean().item(),
            'current_alpha': self.alpha
        }
        