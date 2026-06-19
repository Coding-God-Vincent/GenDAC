import torch
import torch.nn.functional as F
import numpy as np

class PPOopt:

    
    def __init__(
        self,
        actor,
        critic,
        actor_optim,
        critic_optim,
        device,
        gamma= 0.99,
        gae_lambda= 0.95,
        clip_epsilon= 0.2,  # 用在 clip 的那個 epsilon
        epochs= 10,
        entropy_coef= 0.01,  # 控制 Entropy 大小
    ):
        self.actor = actor
        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.entropy_coef = entropy_coef


    # state : np.array with shape (state_dim)
    # action_tanh : torch.tensor with shape (action_dim)
    # log_prob : torch.tensor with shape ()
    # value : float
    def rollout(self, state):
        # shape (1, state_dim)
        state = torch.from_numpy(state).to(dtype= torch.float32, device= self.device).unsqueeze(dim= 0)
        with torch.no_grad():
            # action_tanh : shape (1, action_dim)
            # log_prob : shape (1, 1)
            action_tanh, log_prob = self.actor.sample_action(state= state)
            # value : shape (1, 1)
            value = self.critic(state= state)
        return action_tanh.detach().cpu()[0], log_prob[0, 0].cpu(), value[0].item()
    
    
    # state : np.array with shape (state_dim)
    # raw_action : torch.tensor with shape (action_dim)
    # log_prob : torch.tensor with shape ()
    # value : float
    def rollout_no_tanh(self, state):
        # (1, state_dim)
        state = torch.from_numpy(state).to(dtype= torch.float32, device=self.device).unsqueeze(dim= 0)
        with torch.no_grad():
            raw_action, log_prob = self.actor.sample_action_no_tanh(state= state)
            value = self.critic(state= state)
        return raw_action.detach().cpu()[0], log_prob[0, 0].cpu(), value[0].item()

    
    # last_value : if a trajectory is from timestep 0~T then the last_value = V(s_{T+1}), float
    def update(self, buffer, last_value, batch_size):
        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
        # returns, advantages : torch.tensor with shape (trajectory_length)
        returns, advantages = buffer.compute_GAE(last_value= last_value, gamma= self.gamma, gae_lambda= self.gae_lambda)
        # 一個 trajectory 的資料更新 epochs 次
        for _ in range(self.epochs):
            data_generator = buffer.get_batch(returns= returns, advantages= advantages, batch_size= batch_size)
            # 雖然 data_generator 是回傳一個 tuple，但可以直接用相對位置去接各欄位
            # 以下全都 no grad
            # states[idx] : torch.tensor with shape (batch_size, state_dim)
            # actions[idx] : torch.tensor with shape (batch_size, action_dim)
            # old_log_probs[idx] : torch.tensor with shape (batch_size)
            # returns[idx] : torch.tensor with shape (batch_size)
            # advantages[idx] : torch.tensor with shape (batch_size)
            for state_batch, action_batch, old_log_probs_batch, return_batch, adv_batch in data_generator:
                # 重新評估，產出新的 log_prob
                # current_log_prob : shape (batch_size, 1), with grad
                # entropy : shape (batch_size, 1), with grad
                current_log_prob, entropy = self.actor.evaluate(state= state_batch, action= action_batch)
                # value_pred : shape (batch_size, 1)
                value_pred = self.critic(state= state_batch)
                
                '''Actor Loss'''
                # ratio : shape (batch_size)
                ratio = torch.exp(current_log_prob.squeeze(dim= 1) - old_log_probs_batch)
                # shape ()
                actor_loss = -torch.min(
                    ratio * adv_batch,
                    torch.clamp(input= ratio, min= 1-self.clip_epsilon, max= 1+self.clip_epsilon) * adv_batch
                ).mean()

                '''Critic Loss'''
                # shape ()
                critic_loss = F.mse_loss(input= value_pred.squeeze(dim= 1), target= return_batch)
                
                '''Entropy Loss'''
                # shape ()
                entropy_loss = -entropy.mean()

                total_loss = actor_loss + critic_loss + self.entropy_coef * entropy_loss
                # update
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                
                # 數值通常設 0.5 或 1.0，這能有效防止訓練崩潰
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                self.actor_optim.step()
                self.critic_optim.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        # 回傳平均的 actor, critic, entropy_loss
        steps = self.epochs * (len(buffer.states) // batch_size)
        return total_actor_loss / steps, total_critic_loss / steps, total_entropy_loss / steps
    
    # last_value : if a trajectory is from timestep 0~T then the last_value = V(s_{T+1}), float
    def update_no_tanh(self, buffer, last_value, batch_size):
        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0

        returns, advantages = buffer.compute_GAE(
            last_value= last_value,
            gamma= self.gamma,
            gae_lambda= self.gae_lambda
        )

        for _ in range(self.epochs):
            data_generator = buffer.get_batch(
                returns= returns,
                advantages= advantages,
                batch_size= batch_size
            )

            for state_batch, action_batch, old_log_probs_batch, return_batch, adv_batch in data_generator:
                current_log_prob, entropy = self.actor.evaluate_no_tanh(
                    state= state_batch,
                    action= action_batch
                )

                value_pred = self.critic(state= state_batch)

                ratio = torch.exp(current_log_prob.squeeze(dim= 1) - old_log_probs_batch)

                actor_loss = -torch.min(
                    ratio * adv_batch,
                    torch.clamp(
                        input=ratio,
                        min=1 - self.clip_epsilon,
                        max=1 + self.clip_epsilon
                    ) * adv_batch
                ).mean()

                critic_loss = F.mse_loss(
                    input=value_pred.squeeze(dim=1),
                    target=return_batch
                )

                entropy_loss = -entropy.mean()

                total_loss = actor_loss + critic_loss + self.entropy_coef * entropy_loss

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                self.actor_optim.step()
                self.critic_optim.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_loss.item()

        steps = self.epochs * (len(buffer.states) // batch_size)
        return total_actor_loss / steps, total_critic_loss / steps, total_entropy_loss / steps