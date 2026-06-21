import torch
import torch.nn.functional as F
import numpy as np
from .ReplayBuffer import ReplayBuffer

class SAC_opt:
    

    def __init__(
        self, 
        state_dim, 
        action_dim, 
        actor, 
        critic, 
        target_critic,
        actor_optim,
        critic_optim,
        gamma,  # discount factor
        tau,  # used in soft update
        device,
        alpha_lr= 3e-4,  # 有預設值的參數需要放在沒有預設值參數的後面
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.target_critic.load_state_dict(self.critic.state_dict())  # load critic parameters to target critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.alpha_lr = alpha_lr

        '''Automatic Entropy Tuning (Proposed in the 2nd version of SAC in 2018) : 即把控制 Entropy 影響的參數 (alpha) 用學的，不直接訂死
           作法 : 設定一個最低的 Entropy 標準，若當前的 entropy 很低，就調大 alpha，反之就調小。
        '''
        # 最低 Entropy (希望模型保持的最低隨機性) 設為 -action_dim 是一個經驗法則
        self.target_entropy = -action_dim 
        # 用 log_alpha 而非 alpha 是因為要避免 alpha 為 0，後面在使用時會用 exp 取出。若 alpha 為 0 則沒有意義
        # 初始化 log_alpha = 0，等同於初始 alpha = e^0 = 1
        self.log_alpha = torch.zeros(1, requires_grad= True, device= self.device)  # shape (1)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr= self.alpha_lr)
        self.alpha = self.log_alpha.exp().item()
    
    
    def center_logits(self, action):
        return action - action.mean(dim= -1, keepdim= True)

    
    '''Rollout'''
    # state : np.array with shape (3)
    # output : np.array with shape (3)
    def select_action(self, state):
        state = torch.from_numpy(state).to(dtype= torch.float32, device= self.device).unsqueeze(dim= 0)  # shape (1, state_dim)
        with torch.no_grad():
            # action : tanh(logits by actor), shape (1, action_dim)
            action, _ = self.actor.sample_action(state= state)  
            action = self.center_logits(action= action)
        return action.cpu()[0]  # tensor with shape (action_dim)
    
    
    def select_action_no_tanh(self, state):
        # shape (1, state_dim)
        state = torch.from_numpy(state).to(
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(dim=0)
        with torch.no_grad():
            raw_action, _ = self.actor.sample_action_no_tanh(state=state)
            raw_action = self.center_logits(action= raw_action)
        return raw_action.cpu()[0]  # shape (action_dim)

    
    '''Update'''
    def update(self, buffer, batch_size):
        # state : shape (batch_size, state_dim)
        # action : shape (batch_size, action_dim)
        # reward : shape (batch_size, 1)
        # next_state : shape (batch_size, state_dim)
        state, action, reward, next_state = buffer.sample(batch_size)

        '''Update Critic'''
        # Critic Loss
        with torch.no_grad():
            # next_action : shape (batch_size, action_dim)
            # next_log_prob : future entropy (entropy of \pi(a_(t+1)|s_(t+1))), shape (batch_size, 1)
            next_action, next_log_prob = self.actor.sample_action(next_state)
            next_action = self.center_logits(action= next_action)
            # Q_values : shape (batch_size, 1)
            next_Q_values = self.target_critic.q_min(state= next_state, action= next_action)
            target_Q = reward + self.gamma * (next_Q_values - self.alpha * next_log_prob)
        # current_Q1_values, current_Q2_values : shape (batch_size, 1)
        current_Q1_values, current_Q2_values = self.critic(state= state, action= action)
        critic_loss = F.mse_loss(current_Q1_values, target_Q) + F.mse_loss(current_Q2_values, target_Q)
        # Update Critic Network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        '''Update Actor'''
        # Actor Loss
        # current_action : shape (batch_size, action_dim)
        # current_log_prob : shape (batch_size, 1)
        current_action, current_log_prob = self.actor.sample_action(state)
        current_action = self.center_logits(action= current_action)
        Q_values = self.critic.q_min(state= state, action= current_action)
        actor_loss = (self.alpha * current_log_prob - Q_values).mean()
        # Update Actor Network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        '''Soft Update Target Critic'''
        # 把每一個 critic 參數跟對應的 target_critic 中的參數組成一個 tuple (by zip) 後取出進行 Soft update
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            # .data 取出純數值，因為我們不要梯度
            # copy_ : 代表 in-place，不另外創建一個 tensor
            # copy_(...) 中 ... 為 target critic update 的目標
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        '''Update alpha (by updating log_alpha in order to preventing alpha from being 0)'''
        # alpha loss
        # 加上 detach() 是因為不想讓 alpha_loss 的更新影響到 actor
        alpha_loss = -(self.log_alpha * (current_log_prob + self.target_entropy).detach()).mean()
        # Update alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # update currently used alpha
        self.alpha = self.log_alpha.exp().item()

        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), self.alpha
    
    
    def update_no_tanh(self, buffer, batch_size):
        state, action, reward, next_state = buffer.sample(batch_size)
        # --------------------------------------------------
        # Update Critic
        # --------------------------------------------------
        with torch.no_grad():
            # no_tanh 版本：next_action 是 raw action logits
            next_action, next_log_prob = self.actor.sample_action_no_tanh(next_state)
            next_action = self.center_logits(action= next_action)
            next_Q_values = self.target_critic.q_min(
                state=next_state,
                action=next_action
            )
            target_Q = reward + self.gamma * (next_Q_values - self.alpha * next_log_prob)
        current_Q1_values, current_Q2_values = self.critic(
            state=state,
            action=action
        )
        critic_loss = F.mse_loss(current_Q1_values, target_Q) + F.mse_loss(current_Q2_values, target_Q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # --------------------------------------------------
        # Update Actor
        # --------------------------------------------------
        current_action, current_log_prob = self.actor.sample_action_no_tanh(state)
        current_action = self.center_logits(action= current_action)
        Q_values = self.critic.q_min(
            state=state,
            action=current_action
        )
        actor_loss = (self.alpha * current_log_prob - Q_values).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --------------------------------------------------
        # Soft Update Target Critic
        # --------------------------------------------------
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # --------------------------------------------------
        # Update Alpha
        # --------------------------------------------------
        alpha_loss = -(
            self.log_alpha * (current_log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp().item()

        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), self.alpha

