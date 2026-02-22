import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from Env.env_fixedUE import cellularEnv
from Env.env_movingUE import EnvMove
from tianshou.data import Batch, ReplayBuffer
from pathlib import Path
from pprint import pprint
from Utils.seed import set_seed

# 匯入 MLP ablation 模組 (請確保此路徑與你的資料夾結構相符)
from Utils.MlpAC_utils.Model import GaussianActor, DoubleCritic
from Utils.MlpAC_utils.MlpAC_opt import MlpAC_opt

seeds_fixed = [123, 124, 125, 126, 127]
exps_fixed = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']

seeds_moving = [123, 124, 125, 126, 127]
exps_moving = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']

fixed_or_not = [True, False]

for fixed in fixed_or_not:
    
    if fixed == True : 
        seeds = seeds_fixed
        exps = exps_fixed
    else:
        seeds = seeds_moving
        exps = exps_moving
        
    for i in range(len(seeds)):
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # 環境設定
        set_seed(seed= seeds[i])
        fixed_UE = fixed  # True if using fixed UE env, False if moving UE env
        if fixed_UE: print("\n================================================== fixedUE_env ==================================================\n")
        else: print("\n================================================== movingUE_env ==================================================\n")

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # Log 與圖表路徑設定
        algo_name = 'MlpAC'
        exp_name = exps[i]

        log_file = 'Logs_movingUE_env' if fixed_UE == False else 'Logs_fixedUE_env'
        log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs") / log_file / algo_name / exp_name / 'tensorboard'

        writer = SummaryWriter(log_dir= log_path)

        if fixed_UE: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env") / algo_name / f"{exp_name}"
        else: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env") / algo_name / f"{exp_name}"


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''State Preprocessing'''
        # state : np.array with shape (state_dim)
        # return : np.array with shape (state_dim)
        def state_preprocessing(state):
            log_state = np.log1p(state) 
            return log_state / 10.0  

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # state : np.array with shape (state_dim)
        # return : 
        # action_logits, real_action : np.array with shape (action_dim)
        def get_actions(state, total_band, model, device):
            # shape (1, state_dim)
            state = torch.from_numpy(state).reshape(1, state_dim).to(dtype= torch.float32, device= device)
            with torch.no_grad():
                # shape (1, action_dim)
                action_logit, _ = model(state= state, deterministic= False) 
            # shape (action_dim)
            proportion = torch.nn.functional.softmax(action_logit, dim= 1).cpu().numpy().squeeze()
            real_action = total_band * proportion
            return action_logit.cpu().numpy().squeeze(), real_action

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # qoe : np.array with shape (3)
        # se : np.array with shape (1)
        # return : 
        # utility, reward : np.array with shape (1)
        def cal_reward(qoe, se, qoe_weights, se_weight, reward_clipping= False):
            standard = 0.98  
            standard2 = 0.95  
            utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]  
            if qoe[1] >= standard and qoe[0] >= standard:
                if qoe[2] >= standard2:
                    reward = (np.matmul(qoe_weights, qoe.reshape((3, 1))) + (se_weight / 100.0) * se[0])[0] / 10  
                else:
                    reward = (qoe[2] - standard2) - 0.5  
            else:
                reward = -1.5  - max(0, standard - qoe[0]) - max(0, standard - qoe[1])
            reward = np.array([reward])
            return utility, reward

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        def moving_average(data, window_size):
            data = np.array(data)
            return np.convolve(data, np.ones(window_size) / window_size, mode= 'valid')

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # set the device & some parameters
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ser_cat = ['volte', 'embb_general', 'urllc']

        # training parameters
        state_dim = len(ser_cat)
        action_dim = len(ser_cat)
        actor_lr = 3e-4
        critic_lr = 3e-4
        alpha_lr = 3e-4 # 學習 alpha 的 LR
        buffer_size = 10000 
        batch_size = 32  

        hparams_dict = {
            'actor_lr' : actor_lr,
            'critic_lr' : critic_lr,
            'alpha_lr': alpha_lr,
            'buffer_size' : buffer_size,
            'batch_size' : batch_size,
            'note' : 'MlpAC Ablation Baseline'
        }

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # initialize model & optimizer
        actor = GaussianActor(state_dim= state_dim, action_dim= action_dim).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr= actor_lr)
        critic = DoubleCritic(state_dim= state_dim, action_dim= action_dim).to(device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr= critic_lr)

        # initialize elements
        buffer = ReplayBuffer(size= buffer_size)
        mlp_opt = MlpAC_opt(
            actor= actor,
            actor_optim= actor_optim,
            critic= critic,
            critic_optim= critic_optim,
            device= device,
            state_dim= state_dim,
            action_dim= action_dim,
            alpha_lr= alpha_lr
        )

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # generate the env
        total_band = 10 * 10**6  
        qoe_weights = [1, 1, 1]  
        se_weight = 0.01  
        total_timesteps = 10000  
        learning_windows = 2000  
        dl_mimo = 16  
        UE_no = 100 if fixed_UE else 300

        if fixed_UE: env = cellularEnv(ser_cat= ser_cat, learning_windows= learning_windows, dl_mimo= dl_mimo, UE_max_no= UE_no, hard_scenario= False)
        else: env = EnvMove(UE_max_no= UE_no, ser_prob= np.array([1, 2, 3], dtype= np.float32), learning_windows= learning_windows, dl_mimo= dl_mimo)

        env.countReset()  
        if not fixed_UE: env.user_move()  
        env.activity()  
        observation_packets, observation_bits = env.get_state()  

        QoEs = []
        SEs = []
        Utilities = []
        Rewards = []

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # training loop
        for frame in tqdm(range(1, total_timesteps+1)):
            print(f"\n\n******Episode {frame} :")
            state = state_preprocessing(state= observation_bits)  

            action_logit, real_action = get_actions(state= state, total_band= total_band, model= actor, device= device)
            
            env.band_ser_cat = real_action
            
            for i in range(learning_windows):
                env.scheduling()  
                env.provisioning()  
                env.activity()  

            qoe, se = env.get_reward()
            individual_se, urllc_perfect, urllc_tolerable, urllc_fail, idle_frame = env.eval_get_obs()
            utility, reward = cal_reward(qoe= qoe, se= se, qoe_weights= qoe_weights, se_weight= se_weight, reward_clipping= False)

            QoEs.append(qoe.tolist())  
            SEs.append(se.tolist()[0])  
            Rewards.append(reward.item())  
            Utilities.append(utility.item())

            data = Batch(
                obs= state, 
                act = action_logit,  
                rew = reward.squeeze(),  
                terminated= False,
                truncated= False,
                obs_next= state_preprocessing(env.get_state()[1])  # np.array with shape (3)
            )
            buffer.add(data)
            
            if len(buffer) > batch_size * 3:
                loss = mlp_opt.update(sample_size= batch_size, buffer= buffer)
                pprint(f"loss = {loss}")
                
                # 紀錄包含 Alpha 在內的所有 Loss
                writer.add_scalar(tag= 'loss/actor_loss', scalar_value= loss['actor_loss'], global_step= frame)
                writer.add_scalar(tag= 'loss/critic_loss', scalar_value= loss['critic_loss'], global_step= frame)
                writer.add_scalar(tag= 'loss/alpha_loss', scalar_value= loss['alpha_loss'], global_step= frame)
                writer.add_scalar(tag= 'loss/policy_entropy', scalar_value= loss['policy_entropy'], global_step= frame)
                writer.add_scalar(tag= 'hyperparameter/alpha', scalar_value= loss['current_alpha'], global_step= frame)

            print(f"qoe = {qoe}, se = {float(se[0]):.3f}, reward = {float(reward[0]):.3f}, utility = {float(utility[0]):.3f}")
            
            writer.add_scalar(tag= 'idle_frame', scalar_value= idle_frame, global_step= frame)
            writer.add_scalar(tag= 'pending_packets/volte', scalar_value= env.pending_packets[0], global_step= frame)  
            writer.add_scalar(tag= 'pending_packets/embb_general', scalar_value= env.pending_packets[1], global_step= frame)
            writer.add_scalar(tag= 'pending_packets/urllc', scalar_value= env.pending_packets[2], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/perfect', scalar_value= urllc_perfect, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/tolerable', scalar_value= urllc_tolerable, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail', scalar_value= urllc_fail, global_step= frame)
            writer.add_scalar(tag= 'action/volte', scalar_value= real_action[0], global_step= frame)  
            writer.add_scalar(tag= 'action/embb_general', scalar_value= real_action[1], global_step= frame)
            writer.add_scalar(tag= 'action/urllc', scalar_value= real_action[2], global_step= frame)
            writer.add_scalar(tag= 'observationBits/volte', scalar_value= observation_bits[0], global_step= frame)
            writer.add_scalar(tag= 'observationBits/embb_general', scalar_value= observation_bits[1], global_step= frame)
            writer.add_scalar(tag= 'observationBits/urllc', scalar_value= observation_bits[2], global_step= frame)
            writer.add_scalar(tag= 'qoe/volte', scalar_value= qoe[0], global_step= frame)
            writer.add_scalar(tag= 'qoe/embb_general', scalar_value= qoe[1], global_step= frame)
            writer.add_scalar(tag= 'qoe/urllc', scalar_value= qoe[2], global_step= frame)
            writer.add_scalar(tag= 'se', scalar_value= se[0], global_step= frame)
            writer.add_scalar(tag= 'reward', scalar_value= reward[0], global_step= frame)
            writer.add_scalar(tag= 'utility', scalar_value= utility[0], global_step= frame)
            
            observation_packets, observation_bits = env.get_state()
            env.countReset()
            if not fixed_UE: env.user_move()
            
        writer.add_hparams(hparam_dict= hparams_dict, metric_dict= {})
        print("Complete")

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # Generate Outcome Figures
        qoe_volte = [v for (v, e, u) in QoEs]
        qoe_embb = [e for (v, e, u) in QoEs]
        qoe_urllc = [u for (v, e, u) in QoEs]
        Utilities_ = [u for u in Utilities[1:]]

        ma_qoe_volte = moving_average(qoe_volte, window_size = 200)
        ma_qoe_embb = moving_average(qoe_embb, window_size = 200)
        ma_qoe_urllc = moving_average(qoe_urllc, window_size = 200)
        ma_SE = moving_average(SEs, window_size = 200)
        ma_utility = moving_average(Utilities_, window_size = 200)

        plt.figure(3)
        plt.clf()
        plt.title('QoE')
        plt.xlabel('Episode')
        plt.ylabel('SLA Satisfication Rate')
        plt.plot(ma_qoe_volte)
        plt.plot(ma_qoe_embb)
        plt.plot(ma_qoe_urllc)
        plt.legend(["VoLTE", "Video", "URLLC"])
        plt.savefig(image_path / f"QoE.png")

        plt.figure(4)
        plt.clf()
        plt.title('SE')
        plt.xlabel('Episode')
        plt.ylabel('bits/Hz')
        plt.plot(ma_SE)
        plt.savefig(image_path / f"SE.png")

        plt.figure(5)
        plt.clf()
        plt.title('Utility')
        plt.xlabel("Episode")
        plt.ylabel("utility")
        plt.plot(ma_utility)
        plt.savefig(image_path / f"Utility.png")

        print("Graph Saved")