from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy

from Utils.MlpAC_clamp_rec_utils.ActionTransform import (
    center_action,
    clamp_and_center_action,
)


class MlpAC_opt(BasePolicy):
    """
    Gaussian MLP Actor-Critic with:

    1. Twin Critic
    2. Target Actor / Target Critic
    3. n-step return
    4. Clamp bound
    5. Action centering
    6. Replay-Buffer reconstruction loss

    Actor loss:

        actor_loss
        = policy_loss + recon_param * recon_loss

    where:

        policy_loss
        = -E[min(Q1(s, a), Q2(s, a))]

        recon_loss
        = MSE(center(actor_mean(s)), replay_action)
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        device: str | torch.device,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_step: int = 3,
        action_bound: float = 3.0,
        with_rec_loss: bool = True,
        recon_param: float = 0.5,
        grad_clip_norm: float = 0.5,
        lr_decay: bool = False,
        lr_max_step: int = 10000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if state_dim <= 0:
            raise ValueError(
                f"state_dim must be positive, got {state_dim}"
            )

        if action_dim <= 0:
            raise ValueError(
                f"action_dim must be positive, got {action_dim}"
            )

        if not 0.0 <= gamma <= 1.0:
            raise ValueError(
                f"gamma must be in [0, 1], got {gamma}"
            )

        if not 0.0 < tau <= 1.0:
            raise ValueError(
                f"tau must be in (0, 1], got {tau}"
            )

        if n_step <= 0:
            raise ValueError(
                f"n_step must be positive, got {n_step}"
            )

        if action_bound <= 0:
            raise ValueError(
                f"action_bound must be positive, got {action_bound}"
            )

        if recon_param < 0:
            raise ValueError(
                f"recon_param must be non-negative, got {recon_param}"
            )

        self.actor = actor
        self.actor_optim = actor_optim

        self.actor_target = copy.deepcopy(actor).to(device)
        self.actor_target.eval()

        self.critic = critic
        self.critic_optim = critic_optim

        self.critic_target = copy.deepcopy(critic).to(device)
        self.critic_target.eval()

        # Target networks 只透過 soft update 更新。
        for parameter in self.actor_target.parameters():
            parameter.requires_grad_(False)

        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step

        self.action_bound = float(action_bound)

        self.with_rec_loss = with_rec_loss
        self.recon_param = float(recon_param)

        self.grad_clip_norm = float(grad_clip_norm)

        self.lr_decay = lr_decay

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optim,
                T_max=lr_max_step,
                eta_min=0.0,
            )

            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optim,
                T_max=lr_max_step,
                eta_min=0.0,
            )

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------

    @staticmethod
    def set_requires_grad(
        model: torch.nn.Module,
        requires_grad: bool,
    ) -> None:
        """
        暫時開啟或關閉網路參數的梯度。
        """
        for parameter in model.parameters():
            parameter.requires_grad_(requires_grad)

    def process_action(
        self,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        所有 Actor 產生、且要送入 Critic 的 action，
        都必須使用相同的 Action Transform。
        """
        return clamp_and_center_action(
            raw_action=raw_action,
            bound=self.action_bound,
        )

    # ---------------------------------------------------------------------
    # BasePolicy forward
    # ---------------------------------------------------------------------

    def forward(
        self,
        batch: Batch,
        state: str = "obs",
        model: str = "actor",
        deterministic: bool = False,
        with_logprob: bool = False,
        **kwargs: Any,
    ) -> Batch:
        """
        根據 batch.obs 或 batch.obs_next 產生 action。

        Returns:
            act:
                clamp、中心化與邊界修正後的 action

            raw_act:
                Gaussian Actor 尚未處理的 action

            mean:
                Gaussian distribution mean

            log_prob:
                原始 Gaussian sample 的 log probability
        """
        if state not in {"obs", "obs_next"}:
            raise ValueError(
                f"state must be 'obs' or 'obs_next', got {state}"
            )

        if model not in {"actor", "target_actor"}:
            raise ValueError(
                f"model must be 'actor' or 'target_actor', got {model}"
            )

        state_tensor = to_torch(
            batch[state],
            dtype=torch.float32,
            device=self.device,
        )

        selected_actor = (
            self.actor
            if model == "actor"
            else self.actor_target
        )

        raw_action, mean, log_prob = selected_actor(
            state=state_tensor,
            deterministic=deterministic,
            with_logprob=with_logprob,
        )

        processed_action = self.process_action(
            raw_action=raw_action
        )

        return Batch(
            act=processed_action,
            raw_act=raw_action,
            mean=mean,
            log_prob=log_prob,
            state=state_tensor,
        )

    # ---------------------------------------------------------------------
    # Target Q
    # ---------------------------------------------------------------------

    def td_target_q(
        self,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> torch.Tensor:
        """
        計算 n-step TD target 中最後的 bootstrap Q：

            min_k Q_target_k(
                s_(t+n),
                a_target_(t+n)
            )
        """
        batch = buffer[indices]

        next_state = to_torch(
            batch.obs_next,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            raw_next_action, _, _ = self.actor_target(
                state=next_state,
                deterministic=False,
                with_logprob=False,
            )

            next_action = self.process_action(
                raw_action=raw_next_action
            )

            target_q = self.critic_target.q_min(
                state=next_state,
                action=next_action,
            )

        return target_q

    # ---------------------------------------------------------------------
    # n-step return
    # ---------------------------------------------------------------------

    def process_fn(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> Batch:
        """
        計算：

            r_t
            + gamma r_(t+1)
            + ...
            + gamma^n Q_target(s_(t+n), a_(t+n))

        計算結果會放入 batch.returns。
        """
        return self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self.td_target_q,
            gamma=self.gamma,
            n_step=self.n_step,
        )

    # ---------------------------------------------------------------------
    # Critic update
    # ---------------------------------------------------------------------

    def update_critic(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Replay Buffer 的 batch.act 已經是：

            clamp
            -> center
            -> bound correction

        因此不需要再次處理。
        """
        state = to_torch(
            batch.obs,
            dtype=torch.float32,
            device=self.device,
        )

        replay_action = to_torch(
            batch.act,
            dtype=torch.float32,
            device=self.device,
        )

        td_target = to_torch(
            batch.returns,
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        td_target = td_target.detach()

        current_q1, current_q2 = self.critic(
            state=state,
            action=replay_action,
        )

        critic_loss = (
            F.mse_loss(
                current_q1,
                td_target,
            )
            + F.mse_loss(
                current_q2,
                td_target,
            )
        )

        self.critic_optim.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=self.grad_clip_norm,
        )

        self.critic_optim.step()

        return critic_loss

    # ---------------------------------------------------------------------
    # Actor loss
    # ---------------------------------------------------------------------

    def calculate_actor_losses(
        self,
        batch: Batch,
    ):
        """
        計算：

            policy_loss
            recon_loss
            actor_loss

        Reconstruction branch 使用 Gaussian mean，而非 sample。

        Reconstruction branch 只做中心化，不經過 clamp，
        避免 Actor mean 超過邊界後，reconstruction gradient
        也被 clamp 截斷。
        """
        state = to_torch(
            batch.obs,
            dtype=torch.float32,
            device=self.device,
        )

        replay_action = to_torch(
            batch.act,
            dtype=torch.float32,
            device=self.device,
        )

        # -------------------------------------------------------------
        # 1. Policy loss
        # -------------------------------------------------------------

        raw_policy_action, actor_mean, _ = self.actor(
            state=state,
            deterministic=False,
            with_logprob=False,
        )

        processed_policy_action = self.process_action(
            raw_action=raw_policy_action
        )

        policy_q = self.critic.q_min(
            state=state,
            action=processed_policy_action,
        )

        policy_loss = -policy_q.mean()

        # -------------------------------------------------------------
        # 2. Reconstruction loss
        # -------------------------------------------------------------

        # actor_mean 是同一次 forward 所產生的 Gaussian mean。

        recon_loss = F.mse_loss(
            actor_mean,
            replay_action,
        )

        # -------------------------------------------------------------
        # 3. Total Actor loss
        # -------------------------------------------------------------

        if self.with_rec_loss:
            actor_loss = (
                policy_loss
                + self.recon_param * recon_loss
            )
        else:
            actor_loss = policy_loss

        return (
            actor_loss,
            policy_loss,
            recon_loss,
            raw_policy_action,
            processed_policy_action,
            actor_mean,
        )

    # ---------------------------------------------------------------------
    # Target network update
    # ---------------------------------------------------------------------

    def update_target_networks(self) -> None:
        """
        Soft update：

            target
            = tau * online
            + (1 - tau) * target
        """
        self.soft_update(
            tgt=self.actor_target,
            src=self.actor,
            tau=self.tau,
        )

        self.soft_update(
            tgt=self.critic_target,
            src=self.critic,
            tau=self.tau,
        )

    # ---------------------------------------------------------------------
    # Learn
    # ---------------------------------------------------------------------

    def learn(
        self,
        batch: Batch,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        使用一個 batch 更新：

            1. Twin Critic
            2. Gaussian Actor
            3. Target Actor / Target Critic
        """

        # -------------------------------------------------------------
        # 1. Critic update
        # -------------------------------------------------------------

        critic_loss = self.update_critic(
            batch=batch
        )

        # -------------------------------------------------------------
        # 2. Actor update
        # -------------------------------------------------------------

        # Actor 更新時，不需要計算 Critic 參數的梯度。
        # 但 Critic 對 action 的梯度仍會保留。
        self.set_requires_grad(
            model=self.critic,
            requires_grad=False,
        )

        try:
            (
                actor_loss,
                policy_loss,
                recon_loss,
                raw_policy_action,
                processed_policy_action,
                bc_prediction,
            ) = self.calculate_actor_losses(
                batch=batch
            )

            self.actor_optim.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=self.grad_clip_norm,
            )

            self.actor_optim.step()

        finally:
            self.set_requires_grad(
                model=self.critic,
                requires_grad=True,
            )

        # -------------------------------------------------------------
        # 3. Target update
        # -------------------------------------------------------------

        self.update_target_networks()

        # -------------------------------------------------------------
        # 4. Statistics
        # -------------------------------------------------------------

        with torch.no_grad():
            raw_action_abs_max = (
                raw_policy_action
                .abs()
                .max()
                .item()
            )

            raw_action_abs_mean = (
                raw_policy_action
                .abs()
                .mean()
                .item()
            )

            processed_action_abs_max = (
                processed_policy_action
                .abs()
                .max()
                .item()
            )

            # 中心化後，這個值應該非常接近 0。
            processed_action_mean_abs = (
                processed_policy_action
                .mean(dim=-1)
                .abs()
                .mean()
                .item()
            )

            bc_prediction_abs_max = (
                bc_prediction
                .abs()
                .max()
                .item()
            )

        return {
            "actor_loss": actor_loss.item(),
            "policy_loss": policy_loss.item(),
            "recon_loss": recon_loss.item(),
            "critic_loss": critic_loss.item(),

            "recon_param": float(
                self.recon_param
            ),

            "raw_action_abs_max":
                raw_action_abs_max,

            "raw_action_abs_mean":
                raw_action_abs_mean,

            "processed_action_abs_max":
                processed_action_abs_max,

            "processed_action_mean_abs":
                processed_action_mean_abs,

            "bc_prediction_abs_max":
                bc_prediction_abs_max,
        }

    # ---------------------------------------------------------------------
    # Replay Buffer update
    # ---------------------------------------------------------------------

    def update(
        self,
        sample_size: int,
        buffer: ReplayBuffer,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        從 Replay Buffer 抽取 batch，計算 n-step return，
        接著更新 Actor 與 Critic。
        """
        if buffer is None:
            raise ValueError(
                "buffer cannot be None"
            )

        if len(buffer) < sample_size:
            raise ValueError(
                "Replay Buffer does not contain enough samples: "
                f"len(buffer)={len(buffer)}, "
                f"sample_size={sample_size}"
            )

        batch, indices = buffer.sample(
            sample_size
        )

        batch = self.process_fn(
            batch=batch,
            buffer=buffer,
            indices=indices,
        )

        result = self.learn(
            batch=batch
        )

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return result