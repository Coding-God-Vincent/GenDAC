import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from Env.env_fixedUE import cellularEnv  # GANDDQN 環境 (不考慮使用者移動、考慮 100 人)
from Env.env_movingUE import EnvMove  # LSTM 環境 (考慮使用者移動、考慮 1200 人)
from Utils.Diffusion_utils.diffusion import Diffusion
from Utils.Diffusion_utils.D2AC_opt import D2AC_OPT
from Utils.Diffusion_utils.D2AC_model import GDM, DoubleCritic
from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer, to_torch
from gymnasium.spaces import Discrete, Box  # In order to use BasePolicy
from pathlib import Path
from pprint import pprint
from Utils.seed import set_seed
from Utils.Diffusion_utils.helpers import GaussianNoise
import math


'''Functions'''
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# get current max_action : 從 1 開始每 1/4 就加 1
# step : current step, int
# total_steps : int
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_dynamic_max_action(step, total_steps, qoe_slack, current_success, current_max_action, threshold= 20, tolerance= 0.3):
    # if step < 500 : return 1
    # elif step < 1500 : return 2
    # elif step < 3500 : return 3
    # else: return 4
    # if step < 500 : return 1
    # else : return 2
    # 
    # if step < 500 : return 1.0
    # elif step < 1500 : return 1.0 + (step - 500) / 1000
    # else: return 2.0
    
    # if step < 500 : return 1.0
    # elif step < 1000 : return 1.1
    # elif step < 1500 : return 1.2
    # elif step < 2000 : return 1.3
    # elif step < 2500 : return 1.4
    # elif step < 3000 : return 1.5
    # elif step < 3500 : return 1.6
    # elif step < 4000 : return 1.7
    # elif step < 4500 : return 1.8
    # elif step < 5000 : return 1.9
    # else: return 2.0
    
    # success_streak = current_success
    # if qoe_slack > tolerance: success_streak = 0
    # else: success_streak += 1
    # if success_streak >= threshold and current_max_action < 3.0:
    #     current_max_action = round(current_max_action + 0.1, 1)
    #     success_streak = 0  # next stage
    #     print("Pass ! move to next max_action !")
    
    # return success_streak, current_max_action

    return 3

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# decay the exploration rate in cosin. It's used when exploration_rate_decay is True
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_exploration_rate(step, start_ep, end_ep, start_rate, end_rate):
    if step < start_ep:
        return start_rate
    elif step > end_ep:
        return end_rate
    else:
        ratio = (step - start_ep) / (end_ep - start_ep)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * ratio))
        return end_rate + (start_rate - end_rate) * cosine_decay


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# decay lambda in cosin. To achieve stable learning in the early stage while maintaining high performance in the later stage
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_lambda(current_step, start_step, end_step, start_lambda, end_lambda):
    if current_step < start_step: return start_lambda
    elif current_step > end_step: return end_lambda
    else:
        ratio = (current_step - start_step) / (end_step - start_step)  # 目前進行到哪個部份了
        cosine_decay = 0.5 * (1 + math.cos(math.pi * ratio))
        return end_lambda + (start_lambda - end_lambda) * cosine_decay


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# slack calculator : slack = min{qoe[i] - SLA_threshold | i = 0, 1, 2} 即三個切片贏過 SLA threshold 的程度的最小那個
# slack 越大，探索的越兇。讓模型在贏過很多的時候可以大膽的探索
# qoe : np.array with shape (3)
# SLA_threshold : int
def cal_slack(qoe, SLA_threshold):
    return np.min(qoe - SLA_threshold)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# State Preprocessing : log-scale
# state : np.array with shape (3)
# state2 : avg queue length of each slice of the previous window, np.array with shape (3)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def state_preprocessing(state, 
                        avg_queue_length= None,
                        active_user_no= None,
                        avg_distance= None,
                        use_queue_state= True,  # use avg_queue_length as state
                        use_mobility_state= False,  # use avg_user_no & avg_ue_bs_distance as state
                        active_user_norm= 300.0,
                        distance_norm= 40.0
):
    state_features = []
    # 1. admitted demand/traffic loading
    processed_state = np.log1p(state) / 10  # 1e^9 -> 9*ln(1) ~ 20.7, 正規化後會借於 [0, 10]
    state_features.append(processed_state)
    
    # 2. Queue state : avg_queue_len
    if use_queue_state:
        processed_queue_state = avg_queue_length / 5.0  # avg_queue_length [0, 5]，還是正規化成 [0, 1] 跟 processed_state 量級比較接近。
        state_features.append(processed_queue_state)

    # 3. Mobility state : avg_active_ue_no, avg_active_ue_distance
    if use_mobility_state:
        processed_active_user_no = active_user_no / active_user_norm
        processed_avg_distance = avg_distance / distance_norm
        state_features.append(processed_active_user_no)
        state_features.append(processed_avg_distance)
    
    real_state = np.concatenate(state_features).astype(np.float32)
        
    return real_state  


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# total_band : total bandwidth we have
# logit_low : lower bound of the action_logit
# logit_hight : upper bound of the action_logit
# return : random_logit : np.array with shape (3), real_action : np.array with shape (3)
def get_random_actions(total_band, max_action, logit_low = -0.5, logit_high = 0.5, action_dim= 3, scale= True, action_scale_factor= 3):
    random_logit = np.random.uniform(low=logit_low, high=logit_high, size=(action_dim,)).astype(np.float32).copy()  # np.array with shape (action_dim)
    # 中心化，避免 Critic 還要去分 [1, 1, 1] 跟 [1.3, 1.3, 1.3] 的差別
    random_logit = random_logit - random_logit.mean()  # np.array with shape (action_dim)
    # 找出 abs 後最大的值
    max_abs = np.max(np.abs(random_logit))
    # 算出縮放係數 (使用純 Python 的 min 即可，保證不超過 1.0)
    scale_factor = min(1.0, max_action / (max_abs + 1e-8))
    # 執行縮放
    random_logit = random_logit * scale_factor
    if scale: scaled_random_logit = random_logit * action_scale_factor
    proportion = torch.nn.functional.softmax(torch.from_numpy(scaled_random_logit), dim= 0).numpy()
    real_action = total_band * proportion
    return random_logit, scaled_random_logit, real_action


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# used in generating the outcome figure
# np.convolve(data, kernel= np.ones(window_size) / window_size, mode= 'valid')，用 kernel 掃過整個 data (stride = 1)
# kernel : if window_size = 3, then kernel = [1/3, 1/3, 1/3]. 可以想成是每一個資料所佔的比例
# mode= 'valid'，不做 padding，只對完整的 window 做 moving average
def moving_average(data, window_size):
    data = np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode= 'valid')


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# get current action
# state : np.array with shape (3)
# slack_based_explore : bool, True -> exploration rate adjust with slack dynamically, False -> use static exploration rate or decay in cosin
# exploration_rate_decay : dynamically 
# return : action_logit, scaled_action_logit, real_action : np.array with shape (3)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
noise_generator = GaussianNoise()
def get_actions(state, 
                state_dim,
                total_band, 
                model, 
                device, 
                max_action, 
                exploration_rate_decay, 
                step, 
                start_step, 
                end_step, 
                start_rate, 
                end_rate, 
                slack_based_explore, 
                slack,
                scale,
                action_scale_factor
    ):
    state = torch.from_numpy(state).reshape(1, state_dim).to(dtype= torch.float32, device= device)
    # action_logit : (batch_size, action_dim)
    with torch.no_grad():
        action_logit = model(state= state)
    
    if slack_based_explore:
        if slack > 0.02: sigma = 0.3
        elif slack <= 0.01 and slack > 0: sigma = 0.2
        else: sigma = 0.1
        noise = to_torch(noise_generator.generate(action_logit.shape, sigma= sigma), dtype= torch.float32, device= device)
        action_logit = action_logit + noise
    else:
        if exploration_rate_decay:
            exploration_rate = get_exploration_rate(step= step, start_ep= start_step, end_ep= end_step, start_rate= start_rate, end_rate= end_rate)
        else: exploration_rate = start_rate
        if np.random.rand() < exploration_rate:   
            noise = to_torch(noise_generator.generate(action_logit.shape, sigma= 0.1), dtype= torch.float32, device= device)
            action_logit = action_logit + noise
    
    original_logit = action_logit.clone()

    # # 使用當前動態傳入的 max_action 進行 Clamp
    # action_logit = torch.clamp(action_logit, -max_action, max_action)

    # 中心化，避免 Critic 還要去分 [1, 1, 1] 跟 [1.3, 1.3, 1.3] 的差別
    action_logit = action_logit - action_logit.mean(dim= 1, keepdim= True)  # (batch_size, action_dim)

    # 為了避免中心化後的內容超過 max_action，這邊要再等比例縮放
    # 找出 abs 後最大的那個值
    max_abs = torch.max(torch.abs(action_logit), dim= 1, keepdim= True)[0]
    # 若有超過 max_action，那就要等比例縮放，若沒有就維持原比例
    scale_factor = torch.clamp(max_action / (max_abs + 1e-8), max= 1.0)
    action_logit = action_logit * scale_factor

    if scale: scaled_action_logit = action_logit * action_scale_factor

    proportion = torch.nn.functional.softmax(scaled_action_logit, dim= 1).cpu().numpy().squeeze()
    real_action = total_band * proportion
    return original_logit.cpu().numpy().squeeze(), action_logit.cpu().numpy().squeeze(), scaled_action_logit.cpu().numpy().squeeze(), real_action


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Reward Function
# case1 : satisfy all slices (>= SLA_threshold) -> get reward
# case2 : otherwise -> get penalty
# qoe : np.array with shape (3)
# se : np.array with shape (1)
# return : utility, reward : np.array with shape (1)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# def cal_reward(qoe, se, qoe_weights, se_weight, SLA_threshold= 0.95, reward_clipping= False):
#     utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]
#     if ((qoe[2] >= SLA_threshold) and (qoe[1] >= SLA_threshold) and (qoe[0] >= SLA_threshold)):
#         reward = (np.matmul(qoe_weights, qoe.reshape((3, 1))) + (se_weight / 1.0) * se[0])[0] / 10
#     else:
#         reward = - max(0, SLA_threshold - qoe[0]) - max(0, SLA_threshold - qoe[1]) - max(0, SLA_threshold - qoe[2])
#     reward = np.array([reward])
#     return utility, reward


# reward function3 : exponential gate
def cal_reward(qoe, se, qoe_weights, se_weight, SLA_threshold= 0.95, reward_clipping= False):
    utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]
    # About Qoe
    qoe_score = np.matmul(qoe_weights, qoe.reshape((3, 1)))[0] / 10.0  # int
    qoe_slack = max(0, SLA_threshold - qoe[0]) + max(0, SLA_threshold - qoe[1]) + max(0, SLA_threshold - qoe[2])
    # qoe_penalty = qoe_slack * 2.0
    qoe_penalty = 0.0
    # About SE
    se_base_score = (se_weight * se[0]) / 10.0
    decay = 10
    se_discount = math.exp(-decay * qoe_slack)  # 依照違反程度來決定來指數衰減所得 SE 的好處 (違反越多，衰減越大)
    # final reward 
    reward = qoe_score - qoe_penalty + (se_base_score * se_discount)
    reward = np.array([reward])
    
    return utility, reward, qoe_slack, (se_base_score * se_discount)





'''system env setup'''
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# hyperparameters
fixed_UE = True
dl_mimos = [1, 8]
seeds = [124]
exps = ['exp302', 'exp303']
hard_scenario = False
DDIM = False

'''State Control'''
'''State 1'''
# bits of previous winodw packets of each slice (ever get in queue)
'''State 2'''
# bits of previous winodw packets of each slice (ever get in queue), avg queue length of UEs of each slice
use_queue_state = True
'''State 3'''
use_mobility_state = False
# 新增兩個 mobility 相關的資訊，一個是各切片的 active UE no，另一個是各切片所屬的 active UE 跟 BS 之間的平均距離

# ex: if use_queue_state = True, use_mobility_state = False
# [d1, d2, d3, q1, q2, q3]

# ex: if use_queue_state = True, use_mobility_state = True
# [d1, d2, d3, q1, q2, q3, an1, an2, an3, avgd1, avgd2, avgd3]



for d, dl_mimo in enumerate(dl_mimos):

    for i in range(len(seeds)):

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        set_seed(seed= seeds[i])
        if fixed_UE: print("\n================================================== Fixed_UE env ==================================================\n")
        else: print("\n================================================== Moving_UE env ==================================================\n")
        # 設定圖片 / log 路徑
        algo_name = 'GenDAC'
        exp_name = exps[d]

        log_file = 'Logs_movingUE_env' if fixed_UE == False else 'Logs_fixedUE_env'
        log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs") /log_file / algo_name / exp_name / 'tensorboard'
        # generate log writer
        writer = SummaryWriter(log_dir= log_path)

        # 要看 tensorboard 結果，輸入在 terminal 中他會給你一個網址
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/"algo_name"/"exp_name"/tensorboard"
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/GenDAC/exp21/tensorboard"
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_movingUE_env/GenDAC/exp19/tensorboard"
        # 程式跑下去之後就可以用另一個 terminal 開啟 tensorboard，接著你任何時候想看進度就去點一下 tensorboard 頁面的重置就好了

        if fixed_UE: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/GenDAC") / f"{exp_name}"
        else: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/GenDAC") / f"{exp_name}"
        image_path.mkdir(parents=True, exist_ok=True)

        '''Main'''
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # setup
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # set the device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # env parameters
        ser_cat = ['volte', 'embb_general', 'urllc']
        
        # 依照使用的不同 state 去自動調整 state_dim
        state_block_no = 1
        if use_queue_state: state_block_no += 1
        if use_mobility_state: state_block_no += 2
        state_dim = len(ser_cat) * state_block_no
        action_dim = len(ser_cat)

        # training parameters
        initial_max_action = 3  # will be updated during training (curriculum learning)
        logit_low = -0.5
        logit_high = 0.5
        scale = True  # scale the action or not
        action_scale_factor = 1.0
        total_timesteps = 10000  #  10000 in GAN_DDQN & LSTM_A2C learning_windows (episodes)
        beta_schedule = 'vp'
        if fixed_UE: denoise_step = 3
        else: denoise_step = 7
        actor_lr = 3e-4
        critic_lr = 1e-3
        weight_decay_actor = 1e-4
        weight_decay_critic = 1e-3
        prioritized_replay = False
        buffer_size = 10000
        batch_size = 32
        prior_alpha = 0.4
        prior_beta = 0.4
        start_exploration_rate = 0.1
        end_exploration_rate = 0.1
        start_step = 150
        end_step = 1000
        exploration_rate_decay = False
        SLA_threshold = 0.95
        slack_based_explore = False
        tau = 0.005
        safe_margin = 0.99
        with_action_penalty = False
        initial_lambda = 0.5

        # record training parameters in tensorboard
        note = '動態調整 max_action (1->4)'
        hparams_dict = {
            'denoise step' : denoise_step,
            'actor_lr' : actor_lr,
            'critic_lr' : critic_lr,
            'weight_decay_actor' : weight_decay_actor,
            'buffer_size' : buffer_size,
            'batch_size' : batch_size,
            'note' : note
        }


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # generate models
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        gdm = GDM(state_dim= state_dim, action_dim= action_dim)
        actor = Diffusion(
            state_dim= state_dim,
            action_dim= action_dim,
            model= gdm,
            max_action= initial_max_action,
            beta_schedule= beta_schedule,
            denoise_steps= denoise_step,
            # 用 True 的原因為 : (原始 DDPM 也是這樣)
            # 每一步都 clamp -> 越低的機率出現那種超大的值 -> 被 clamp 的機率越低 -> 學習越穩定。
            clip_denoised= True,  # True -> 中間的每一步的 x_0 都會被 clamp 掉
            device= device,
            DDIM= DDIM
        ).to(device= device)
        actor_optim = torch.optim.AdamW(
            # Diffusion inherits nn.Module, so actor.parameters() will be redirect to the parameters of all nn.Modules included in actor
            params= actor.parameters(),  
            lr= actor_lr,
            weight_decay= weight_decay_actor  # 讓參數慢慢趨近於 0，避免數值爆炸
        )
        scheduler_actor = torch.optim.lr_scheduler.LinearLR(actor_optim, start_factor= 1.0, end_factor= 0.1, total_iters= total_timesteps)

        critic = DoubleCritic(state_dim= state_dim, action_dim= action_dim).to(device= device)
        critic_optim = torch.optim.AdamW(
            params= critic.parameters(),
            lr= critic_lr,
            weight_decay= weight_decay_critic
        )
        scheduler_critic = torch.optim.lr_scheduler.LinearLR(critic_optim, start_factor= 1.0, end_factor= 0.1, total_iters= total_timesteps)
        

        # generate the ReplayBuffer
        if prioritized_replay: 
            buffer = PrioritizedReplayBuffer(
                size= buffer_size,
                # used to control the strength of the prioritization (alpha = 0 : uniform, alpha = 1 : complete prioritized)
                alpha= prior_alpha,
                # used to control the strength of revision of the sampling bias
                beta= prior_beta
            )
        else: buffer = ReplayBuffer(size= buffer_size)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # generate an instance of D2AC_OPT to handle the update of the model
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        fake_action_space = Discrete(3)
        fake_action_space = Box(low= -1, high= 1, shape= (3,))
        d2ac_opt = D2AC_OPT(
            state_dim= state_dim,
            action_dim= action_dim,
            actor= actor,
            actor_optim= actor_optim,
            critic= critic,
            critic_optim= critic_optim,
            device= device,
            n_steps= 3,  
            with_rec_loss= True,
            recon_param= initial_lambda,
            lr_decay= False,
            max_action= initial_max_action,
            tau= tau,
            safe_margin= safe_margin,
            with_action_penalty= with_action_penalty,
            # 以下參數會放在 **kwargs，放一些用不到但 BasePolicy 規定要放的參數
            action_space= fake_action_space
        )

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # generate the env
        # ser_cat = ['volte', 'embb_general', 'urllc']
        if hard_scenario: total_band = 20 * 10**6  # 20MHz (original 10 MHz)
        else: total_band = 10 * 10**6
        # J = \alpha * SE + \betas * SSRs
        qoe_weights = [1, 1, 1]  # \betas
        se_weight = 0.01  # \alpha (原論文設定為 0.01)
        learning_windows = 2000  # 1 learning window (episode) = 2000 timeslots
        prefill_steps = 3 * batch_size
        if hard_scenario: dl_mimo = 3  # 原本是 64
        else: dl_mimo = dl_mimo
        UE_no = 100 if fixed_UE else 300
        if fixed_UE: env = cellularEnv(ser_cat= ser_cat, learning_windows= learning_windows, dl_mimo= dl_mimo, UE_max_no= UE_no, hard_scenario= hard_scenario, schedu_method= 'round_robin_reuse_rem')
        else: env = EnvMove(UE_max_no= UE_no, ser_prob= np.array([1, 2, 3], dtype= np.float32), learning_windows= learning_windows, dl_mimo= dl_mimo, hard_scenario= hard_scenario)
        env.countReset()  # reset 所有計數器
        if not fixed_UE: env.user_move()  # user move in LSTM-A2C env
        env.activity()  # 所有 UE 開始根據其網路切片產生封包
        # observation_packets : total packets of each NSs, np.array with shape (3)
        # observation_bits : total bits of each NSs, np.array with shape (3)
        # avg_queue_length_of_each_slices : 前一個 window 各切片所有 UE 的平均 Queue Length, np.array with shape (3)
        observation_packets, observation_bits, avg_queue_length_of_each_slices = env.get_state2()  
        active_user_no_of_each_slices, avg_distance_of_each_slices = env.get_mobility_state()

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # recording lists
        QoEs = []
        SEs = []
        Utilities = []
        Rewards = []
        Observations = []
        Actor_losses = []
        Critic_losses = []


        '''Training Procedure'''
        slack = 0.0

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # Stage1 : Prefill 
        # print("=================================Prefilliing=================================")
        # for ps in range(prefill_steps):
        #     print(f"\nPrefill step : {ps}")
        #     state = state_preprocessing(state= observation_bits)
        #     action_logit, scaled_random_logit, real_action = get_random_actions(
        #         total_band= total_band,
        #         max_action= initial_max_action,
        #         logit_low= logit_low,
        #         logit_high= logit_high,
        #         action_dim= action_dim,
        #         scale= scale,
        #         action_scale_factor= action_scale_factor
        #     )
        #     print(f"action logit = {action_logit}, scaled action logit = {scaled_random_logit}, proportion = {real_action / 10000000}")
        #     env.band_ser_cat = real_action
        #     # 2000 slots in 1 learning window
        #     for i in range(learning_windows):
        #         env.scheduling()  # do lower-level allocation every timeslots
        #         env.provisioning()  # evaluate the SE & SSR of the current timeslot
        #         env.activity()  # assign readtime & generate packet according to the readtime

        #     dropped_packets = env.eval_get_obs3()  # np.array with shape (3) (volte / embb/ urllc)
            
        #     # qoe : np.array with shape (3)
        #     # se : np.array with shape (1)
        #     qoe, se = env.get_reward()
        #     # utility, reward : np.array with shape (1)
        #     utility, reward, qoe_slack, _ = cal_reward(
        #         qoe= qoe,
        #         se= se,
        #         qoe_weights= qoe_weights,
        #         se_weight= se_weight,
        #         SLA_threshold= SLA_threshold,
        #         reward_clipping= False
        #     )
        #     # Record the values of the current learning window
        #     QoEs.append(qoe.tolist())  # qoe.tolist() -> [qoe1, qoe2, qoe3]
        #     SEs.append(se.tolist()[0])  # se.tolist() -> [se]
        #     Rewards.append(reward.item())  
        #     Utilities.append(utility.item())
            
        #     next_observation_packets, next_observation_bits = env.get_state()
        #     obs_next = state_preprocessing(next_observation_bits)

        #     data = Batch(
        #         obs= state,
        #         act= action_logit,
        #         rew= reward.squeeze(),
        #         terminated= False,
        #         truncated= False,
        #         obs_next= obs_next
        #     )
        #     buffer.add(data)
            
        #     # print the outcome of the current learning window
        #     print(f"qoe = {qoe}, se = {float(se[0]):.3f}, reward = {float(reward[0]):.3f}, utility = {float(utility[0]):.3f}")
            
        #     writer.add_scalar(tag= 'pending_packets/volte', scalar_value= env.pending_packets[0], global_step= ps)  # 每一個 window 分完後各網路切片還剩下多少待傳的 buffer
        #     writer.add_scalar(tag= 'pending_packets/embb_general', scalar_value= env.pending_packets[1], global_step= ps)
        #     writer.add_scalar(tag= 'pending_packets/urllc', scalar_value= env.pending_packets[2], global_step= ps)
        #     writer.add_scalar(tag= 'action/volte', scalar_value= real_action[0], global_step= ps)  # 分配比例
        #     writer.add_scalar(tag= 'action/embb_general', scalar_value= real_action[1], global_step= ps)
        #     writer.add_scalar(tag= 'action/urllc', scalar_value= real_action[2], global_step= ps)
        #     writer.add_scalar(tag= 'observationBits/volte', scalar_value= observation_bits[0], global_step= ps)
        #     writer.add_scalar(tag= 'observationBits/embb_general', scalar_value= observation_bits[1], global_step= ps)
        #     writer.add_scalar(tag= 'observationBits/urllc', scalar_value= observation_bits[2], global_step= ps)
        #     writer.add_scalar(tag= 'observationPackets/volte', scalar_value= observation_packets[0], global_step= ps)
        #     writer.add_scalar(tag= 'observationPackets/embb_general', scalar_value= observation_packets[1], global_step= ps)
        #     writer.add_scalar(tag= 'observationPackets/urllc', scalar_value= observation_packets[2], global_step= ps)
        #     writer.add_scalar(tag= 'qoe/volte', scalar_value= qoe[0], global_step= ps)
        #     writer.add_scalar(tag= 'qoe/embb_general', scalar_value= qoe[1], global_step= ps)
        #     writer.add_scalar(tag= 'qoe/urllc', scalar_value= qoe[2], global_step= ps)
        #     writer.add_scalar(tag= 'se', scalar_value= se[0], global_step= ps)
        #     writer.add_scalar(tag= 'reward', scalar_value= reward[0], global_step= ps)
        #     writer.add_scalar(tag= 'utility', scalar_value= utility[0], global_step= ps)
        #     writer.add_scalar(tag= 'dropped_packet/volte', scalar_value= dropped_packets[0], global_step= ps)
        #     writer.add_scalar(tag= 'dropped_packet/embb_general', scalar_value= dropped_packets[1], global_step= ps)
        #     writer.add_scalar(tag= 'dropped_packet/urllc', scalar_value= dropped_packets[2], global_step= ps)
        #     writer.add_scalar(tag= 'max_action', scalar_value= initial_max_action, global_step= ps)
        #     writer.add_scalar(tag= 'action_logit/volte', scalar_value= action_logit[0], global_step= ps)
        #     writer.add_scalar(tag= 'action_logit/embb_general', scalar_value= action_logit[1], global_step= ps)
        #     writer.add_scalar(tag= 'action_logit/urllc', scalar_value= action_logit[2], global_step= ps)

            
        #     observation_packets, observation_bits = next_observation_packets, next_observation_bits
            
        #     # reset all counters after each learning window
        #     env.countReset()

        #     # if using the env of LSTM-A2C then move the users
        #     if not fixed_UE: env.user_move()

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # Stage2 : Training
        current_success = 0
        qoe_slack = 0
        current_max_action = initial_max_action
        # for frame in tqdm(range(prefill_steps, total_timesteps)):
        for frame in tqdm(range(0, total_timesteps)):
            
            print(f"\n\n******Episode {frame} :")

            # Curicculum Learning : Adjust max_action dynamically
            # calculate the current max_action
            current_max_action = get_dynamic_max_action(
                step= frame, 
                total_steps= total_timesteps, 
                qoe_slack= qoe_slack, 
                current_success= current_success, 
                current_max_action= current_max_action
            )
            # modify max action in created instances
            # # 1. diffusion.py 中有 max_action 屬性
            # actor.max_action = current_max_action
            # # d2ac_opt.py 中有建立 target_actor，那也有 max_action
            # d2ac_opt.target_actor.max_action = current_max_action
            # # 2. D2AC_opt.py 中有 max_action 屬性
            # d2ac_opt.max_action = current_max_action

            # calculate the current lambda
            current_lambda = get_lambda(
                current_step= frame,
                start_step= batch_size * 3,
                end_step= 6000,
                start_lambda= initial_lambda,
                end_lambda= 0.001
            )
            # current_lambda = initial_lambda
            
            # modify lambda in created instances
            d2ac_opt.recon_param = current_lambda

            # state is the loading (no. of packets) of each NS of the previous learning window
            state = state_preprocessing(state= observation_bits, 
                                        avg_queue_length= avg_queue_length_of_each_slices,
                                        active_user_no= active_user_no_of_each_slices,
                                        avg_distance= avg_distance_of_each_slices,
                                        use_queue_state= use_queue_state,
                                        use_mobility_state= use_mobility_state,
                                        active_user_norm= float(UE_no),
                                        distance_norm= 40
            )  
            # print(f"observation_packets = {observation_packets}, observation_bits = {observation_bits}")
            print(f"state = {state}")  # np.array with shape (ser_cat * 2)

            # action_logit : Actor 輸出 torch.tensor with shape (batch_size(1), action_dim), values are within the range(-1, 1)
            # real_action : 將 logit 轉為真實動作，即各網路切片的分配到的頻寬 (Hz)。np.array with shape (3)
            origianl_logit, action_logit, scaled_action_logit, real_action = get_actions(
                state= state, 
                state_dim= state_dim,
                total_band= total_band, 
                model= actor, 
                device= device, 
                max_action= current_max_action,
                exploration_rate_decay= exploration_rate_decay,
                step= frame,
                start_step= start_step,
                end_step= end_step,
                start_rate= start_exploration_rate,
                end_rate= end_exploration_rate,
                slack_based_explore= slack_based_explore,
                slack= slack,
                scale= scale,
                action_scale_factor= action_scale_factor
            )
            print(f"original_logits = {origianl_logit}, action_logits = {action_logit}")
            print(f"scaled_action_logit = {scaled_action_logit}, proportion = {real_action / 10000000}")
            # print(f"action_logit = {action_logit}, real action = {real_action}")
            # print(f"action = {real_action}")
            
            # assign to the env.
            env.band_ser_cat = real_action
            # print(env.band_ser_cat)  # ex: [3442405.76028824 3145710.52789688 3411883.71181488]
            
            # 2000 slots in 1 learning window
            for _ in range(learning_windows):
                env.scheduling()  # do lower-level allocation every timeslots
                env.provisioning()  # evaluate the SE & SSR of the current timeslot
                env.activity()  # assign readtime & generate packet according to the readtime
                env.record_queue_length()
                
            
            # calculate the reward of the current learning window
            # qoe : np.array with shape (3)
            # se : np.array with shape (1)
            qoe, se = env.get_reward()
            

            # calculate the individual se of each network slices of the current learning window
            # indivifual_se : np.array with shape (3)
            # urllc_perfect, tolerable, fail : packet count categorized by latency for transmitted URLLC traffic of the current learning window, int
            individual_se, urllc_perfect, urllc_tolerable, urllc_fail, idle_frame = env.eval_get_obs()
            # 一個 window 中各切片有幾個 slot 是完全沒有 active user (在 BS 內且有封包要傳) 
            volte_UE_slot, embb_UE_slot, urllc_UE_slot, urllc_violate_packet_size = env.eval_get_obs2()
            # 看一個 window 中各切片有多少 packets 被 dropped 掉
            dropped_packets = env.eval_get_obs3()  # np.array with shape (3) (volte / embb/ urllc)
            
            # use qoe & se to calculate utility as a reward
            # utility = \alpha * SE + (\betas * SSRs).sum()
            # utility, reward : np.array with shape (1)
            utility, reward, qoe_slack, se_part = cal_reward(qoe= qoe, se= se, qoe_weights= qoe_weights, se_weight= se_weight, SLA_threshold= SLA_threshold, reward_clipping= False)

            # use qoe & SLA_threshold to calculate slack
            slack = cal_slack(qoe= qoe, SLA_threshold= SLA_threshold)

            # Record the values of the current learning window
            QoEs.append(qoe.tolist())  # qoe.tolist() -> [qoe1, qoe2, qoe3]
            SEs.append(se.tolist()[0])  # se.tolist() -> [se]
            Rewards.append(reward.item())  
            Utilities.append(utility.item())
            
            next_observation_packets, next_observation_bits, next_avg_queue_length = env.get_state2()
            next_active_user_no, next_avg_distance = env.get_mobility_state()
            obs_next = state_preprocessing(
                state= next_observation_bits,
                avg_queue_length= next_avg_queue_length,
                active_user_no= next_active_user_no,
                avg_distance= next_avg_distance,
                use_queue_state= use_queue_state,
                use_mobility_state= use_mobility_state,
                active_user_norm= float(UE_no),
                distance_norm= 40.0
            )

            # store the experience to the ReplayBuffer
            data = Batch(
                obs= state,  # np.array with shape (6)
                act = action_logit,  # np.array with shape (3)
                rew = reward.squeeze(),  # int
                terminated= False,
                truncated= False,
                obs_next= obs_next  # np.array with shape (3)
            )
            buffer.add(data)
            
            # update the model after warming up
            if len(buffer) >= batch_size * 3:
                loss, observe_values = d2ac_opt.update(sample_size= batch_size, buffer= buffer)
                pprint(f"loss = {loss}")
                # adjust learning rate
                # scheduler_actor.step()
                # scheduler_critic.step()
                writer.add_scalar(tag= 'loss/actor_loss', scalar_value= loss['actor_loss'].item(), global_step= frame)
                writer.add_scalar(tag= 'loss/policy_loss', scalar_value= loss['policy_loss'].item(), global_step= frame)
                writer.add_scalar(tag= 'loss/recon_loss', scalar_value= loss['recon_loss'].item(), global_step= frame)
                writer.add_scalar(tag= 'loss/critic_loss', scalar_value= loss['critic_loss'].item(), global_step= frame)
                writer.add_scalar(tag= 'loss/action_penalty', scalar_value= loss['action_penalty'].item(), global_step= frame)
                # writer.add_scalar(tag= 'unclampped_logits/absmin', scalar_value= observe_values['unclampped_logits_absmin'], global_step= frame)
                # writer.add_scalar(tag= 'unclampped_logits/max', scalar_value= observe_values['unclampped_logits_max'], global_step= frame)
                # writer.add_scalar(tag= 'unclampped_logits/absmean', scalar_value= observe_values['unclampped_logits_absmean'], global_step= frame)
                writer.add_scalar(tag= 'unclampped_logits_legal_rate', scalar_value= observe_values['unclampped_logits_legal_rates'], global_step= frame)
                writer.add_scalar(tag= 'grad/grad_norm_policy', scalar_value= loss['grad_norm_policy'], global_step= frame)
                writer.add_scalar(tag= 'grad/grad_norm_rec', scalar_value= loss['grad_norm_rec'], global_step= frame)
                writer.add_scalar(tag= 'grad/grad_norm_ap', scalar_value= loss['grad_norm_ap'], global_step= frame)
                writer.add_scalar(tag= 'grad/grad_cov_policy', scalar_value= loss['grad_cov_policy'], global_step= frame)
                writer.add_scalar(tag= 'grad/grad_cov_rec', scalar_value= loss['grad_cov_rec'], global_step= frame)
                writer.add_scalar(tag= 'grad/grad_cov_ap', scalar_value= loss['grad_cov_ap'], global_step= frame)

            # Actor_losses.append(loss['actor_loss'].item())
            # Critic_losses.append(loss['critic_loss'].item())

            # print the outcome of the current learning window
            print(f"qoe = {qoe}, se = {float(se[0]):.3f}, reward = {float(reward[0]):.3f}, utility = {float(utility[0]):.3f}, se_part = {float(se_part):.3f}")
            writer.add_scalar(tag= 'idle_frame/urllc', scalar_value= urllc_UE_slot, global_step= frame)
            writer.add_scalar(tag= 'idle_frame/volte', scalar_value= volte_UE_slot, global_step= frame)
            writer.add_scalar(tag= 'idle_frame/embb', scalar_value= embb_UE_slot, global_step= frame)
            writer.add_scalar(tag= 'idle_frame', scalar_value= idle_frame, global_step= frame)
            writer.add_scalar(tag= 'pending_packets/volte', scalar_value= env.pending_packets[0], global_step= frame)  # 每一個 window 分完後各網路切片還剩下多少待傳的 buffer
            writer.add_scalar(tag= 'pending_packets/embb_general', scalar_value= env.pending_packets[1], global_step= frame)
            writer.add_scalar(tag= 'pending_packets/urllc', scalar_value= env.pending_packets[2], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/perfect', scalar_value= urllc_perfect, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/tolerable', scalar_value= urllc_tolerable, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail', scalar_value= urllc_fail, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail_6.4KB', scalar_value= urllc_violate_packet_size[0], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail_12.8KB', scalar_value= urllc_violate_packet_size[1], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail_19.2KB', scalar_value= urllc_violate_packet_size[2], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail_25.6KB', scalar_value= urllc_violate_packet_size[3], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail_32KB', scalar_value= urllc_violate_packet_size[4], global_step= frame)
            writer.add_scalar(tag= 'action/volte', scalar_value= real_action[0], global_step= frame)  # 分配比例
            writer.add_scalar(tag= 'action/embb_general', scalar_value= real_action[1], global_step= frame)
            writer.add_scalar(tag= 'action/urllc', scalar_value= real_action[2], global_step= frame)
            writer.add_scalar(tag= 'observationBits/volte', scalar_value= observation_bits[0], global_step= frame)
            writer.add_scalar(tag= 'observationBits/embb_general', scalar_value= observation_bits[1], global_step= frame)
            writer.add_scalar(tag= 'observationBits/urllc', scalar_value= observation_bits[2], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/volte', scalar_value= observation_packets[0], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/embb_general', scalar_value= observation_packets[1], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/urllc', scalar_value= observation_packets[2], global_step= frame)
            writer.add_scalar(tag= 'qoe/volte', scalar_value= qoe[0], global_step= frame)
            writer.add_scalar(tag= 'qoe/embb_general', scalar_value= qoe[1], global_step= frame)
            writer.add_scalar(tag= 'qoe/urllc', scalar_value= qoe[2], global_step= frame)
            writer.add_scalar(tag= 'se', scalar_value= se[0], global_step= frame)
            writer.add_scalar(tag= 'individual_se/volte', scalar_value= individual_se[0], global_step= frame)
            writer.add_scalar(tag= 'individual_se/embb_general', scalar_value= individual_se[1], global_step= frame)
            writer.add_scalar(tag= 'individual_se/urllc', scalar_value= individual_se[2], global_step= frame)
            writer.add_scalar(tag= 'reward', scalar_value= reward[0], global_step= frame)
            writer.add_scalar(tag= 'utility', scalar_value= utility[0], global_step= frame)
            writer.add_scalar(tag= 'dropped_packet/volte', scalar_value= dropped_packets[0], global_step= frame)
            writer.add_scalar(tag= 'dropped_packet/embb_general', scalar_value= dropped_packets[1], global_step= frame)
            writer.add_scalar(tag= 'dropped_packet/urllc', scalar_value= dropped_packets[2], global_step= frame)
            writer.add_scalar(tag= 'max_action', scalar_value= current_max_action, global_step= frame)
            writer.add_scalar(tag= 'lambda', scalar_value= current_lambda, global_step= frame)
            writer.add_scalar(tag= 'action_logit/volte', scalar_value= action_logit[0], global_step= frame)
            writer.add_scalar(tag= 'action_logit/embb_general', scalar_value= action_logit[1], global_step= frame)
            writer.add_scalar(tag= 'action_logit/urllc', scalar_value= action_logit[2], global_step= frame)
            writer.add_scalar(tag= 'avg_queue_length/volte', scalar_value= avg_queue_length_of_each_slices[0], global_step= frame)
            writer.add_scalar(tag= 'avg_queue_length/embb', scalar_value= avg_queue_length_of_each_slices[1], global_step= frame)
            writer.add_scalar(tag= 'avg_queue_length/urllc', scalar_value= avg_queue_length_of_each_slices[2], global_step= frame)
            writer.add_scalar(tag= 'active_user_no/volte', scalar_value= active_user_no_of_each_slices[0], global_step=frame)
            writer.add_scalar(tag= 'active_user_no/embb', scalar_value= active_user_no_of_each_slices[1], global_step=frame)
            writer.add_scalar(tag= 'active_user_no/urllc', scalar_value= active_user_no_of_each_slices[2], global_step=frame)
            writer.add_scalar(tag= 'avg_distance/volte', scalar_value= avg_distance_of_each_slices[0], global_step=frame)
            writer.add_scalar(tag= 'avg_distance/embb', scalar_value= avg_distance_of_each_slices[1], global_step=frame)
            writer.add_scalar(tag= 'avg_distance/urllc', scalar_value= avg_distance_of_each_slices[2], global_step=frame)
            
            # update current state variables for next decision window
            observation_packets = next_observation_packets
            observation_bits = next_observation_bits
            avg_queue_length_of_each_slices = next_avg_queue_length
            active_user_no_of_each_slices = next_active_user_no
            avg_distance_of_each_slices = next_avg_distance

            # reset all counters after each learning window
            env.countReset()

            # if using the env of LSTM-A2C then move the users
            if not fixed_UE: env.user_move()
            

        metric_dict = {}
        writer.add_hparams(hparam_dict= hparams_dict, metric_dict= metric_dict)

        print("Complete")

        # 存下訓練好的參數以供後續產圖
        if fixed_UE:
            torch.save(critic.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/fixed_UE/6_algos/GenDAC/critic_weights.pth')
            torch.save(gdm.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/fixed_UE/6_algos/GenDAC/gdm_weights.pth')
        else:
            torch.save(critic.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/movingUE/6_algos/GenDAC/critic_weights.pth')
            torch.save(gdm.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/movingUE/6_algos/GenDAC/gdm_weights.pth')


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Generate Outcome Figures'''
        qoe_volte = [v for (v, e, u) in QoEs]
        qoe_embb = [e for (v, e, u) in QoEs]
        qoe_urllc = [u for (v, e, u) in QoEs]

        ma_qoe_volte = moving_average(qoe_volte, window_size = 200)
        ma_qoe_embb = moving_average(qoe_embb, window_size = 200)
        ma_qoe_urllc = moving_average(qoe_urllc, window_size = 200)
        ma_SE = moving_average(SEs, window_size = 200)
        ma_utility = moving_average(Utilities, window_size = 200)

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

        writer.close()