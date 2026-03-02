import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import copy
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy

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
        n_step = 3,
        lr_decay= False,
        lr_max_step= 1000,
        **kwargs : any
    ):
        super().__init__(**kwargs)
        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_target = copy.deepcopy(actor)
        self.actor_target.eval()
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_target = copy.deepcopy(critic)
        self.critic_target.eval()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.lr_decay = lr_decay
        
        # Auto-Tuning Alpha (2018 SAC) : proportion of the entropy
        # alpha must be positive so we use log-exp trick here as well
        # target_entropy is set to -dim(action) ususally (experience)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad= True, device= device)
        self.alpha_optim = torch.optim.Adam(params= [self.log_alpha], lr= alpha_lr)
        self.alpha = self.log_alpha.exp().item()  # int
        
        # if we want to decay the lr, use CosineAnnealingLR
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optim, T_max= lr_max_step, eta_min= 0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optim, T_max= lr_max_step, eta_min= 0.)
        
        
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
        model : str = 'actor',
        **kwargs : any
    ):
        # shape (batch_size, state_dim)
        state = to_torch(batch[state], dtype= torch.float32, device= self.device)
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
        batch = buffer[indices]  # type : batch, len(batch_size)
        # shape (batch_size, state_dim)
        next_state = to_torch(batch.obs_next, dtype= torch.float32, device= self.device)
        with torch.no_grad():
            next_action, next_log_prob = self.actor_target(state= next_state)
            alpha = self.log_alpha.exp().detach()
            # shape (batch_size, 1)
            target_Q = self.critic_target.q_min(state= next_state, action= next_action) - alpha * next_log_prob
        return target_Q
    
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 Batch，算出其中各資料的 TD Target。ex: r(t) + \gamma*r(t+1) + \gamma^2*r(t+2) + ... + \gamma^(n-1)*r(t+n-1) + \gamma^(n)*min{ Q^k(s(t+n), a_target(t+n) }
    # 最後將算出的各資料的 TD_target 放入傳入的那個 Batch.returns。回傳傳入的那個 Batch
    # 這個 batch 是一個新的 Batch，不會再回到 ReplayBuffer。所以 return 不用是 np.array 形式，他是 torch.tensor，且是有梯度的。
    # 此函式所回傳的 batch 是複製原本的 batch 的所有資料後再加一個欄位 (returns)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    '''essential in BasePolicy'''
    def process_fn(
        self, 
        batch : Batch,  # batch data
        buffer : ReplayBuffer,  # replay buffer, used to calculate the n-step return (rewards are stored in order in the replay buffer)
        indices : np.ndarray  # 傳入的 batch 於 replay buffer 中的 indices。shape (batch_size)
    ):
        return self.compute_nstep_return(
            batch= batch,
            buffer= buffer,
            indices= indices,
            target_q_fn= self.td_target_q,
            gamma= self.gamma,
            n_step= self.n_step,
        )
    
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # soft update target actor and target Twin Q network
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def update_target_networks(self):
        self.soft_update(tgt= self.actor_target, src= self.actor, tau= self.tau)
        self.soft_update(tgt= self.critic_target, src= self.critic, tau= self.tau)
        
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 batch 的資料並更新所有網路的更新，最後回傳該 batch 的 Actor loss & Critic loss (in dict)
    # batch : extracted batch data (已經經過 process_fn，batch 中的 returns 欄位已經是算好的 td_target)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    '''essential in BasePolicy'''
    def learn(self, batch : Batch):
        # shape (batch_size, state_dim)
        state = to_torch(batch.obs, dtype= torch.float32, device= self.device)
        # shape (batch_size, action_dim)
        action = to_torch(batch.act, dtype= torch.float32, device= self.device)
        # shape (batch_size, 1)
        td_target = to_torch(batch.returns, dtype= torch.float32, device= self.device).view(-1, 1)
        alpha = self.log_alpha.exp()
        
        '''update critic'''
        current_q1, current_q2 = self.critic(state= state, action= action)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        '''update actor'''
        current_action, current_log_prob = self.actor(state= state)
        currentQ = self.critic.q_min(state= state, action= current_action)
        actor_loss = (alpha.detach() * current_log_prob - currentQ).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        '''update alpha'''
        alpha_loss = -(self.log_alpha * (current_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()
        
        '''soft update target networks'''
        self.update_target_networks()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'policy_entropy': -current_log_prob.mean().item(),
            'current_alpha': self.alpha
        }
        
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 ReplayBuffer，從該 buffer 中抽取一個 batch 的資料來更新一遍所有的 networks
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    '''essential in BasePolicy'''
    def update(
        self, 
        sample_size : int, 
        buffer : ReplayBuffer
    ):
        batch, indices = buffer.sample(sample_size)
        # compute TD_target of each data in the batch
        batch = self.process_fn(batch= batch, buffer= buffer, indices= indices)
        # update each network via extracted batch
        result = self.learn(batch= batch)
        # once update the network update the lr
        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
        # return actor loss & critic loss via dict & observe values
        return result