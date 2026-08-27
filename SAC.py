import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from Env.env_fixedUE import cellularEnv
from Env.env_movingUE import EnvMove
from Utils.SAC_utils import Model, ReplayBuffer, SACopt
from Utils.seed import set_seed
from pprint import pprint
import math


'''SAC 作法硬傷
    若直接使用 SAC 那一套，只不過因為這邊是分配資源場景所以我在 tanh 輸出後直接硬套上一個 Softmax，讓其總合為 1。
    這樣做最簡單好改但也會有一個硬傷 : 就是他沒辦法去嘗試很極端的資源分配。
    ex : 假設經過 tanh 之後輸出 : [1, -1, -1] (這已經是最極端的分配的)，那這時候經過 softmax 後會輸出 : [0.787, 0.106, 0.106]
    要解決這個問題最常用的做法就是對 tanh 輸出後的數值最一個放大 : 同乘一個 tau，這邊建議設 5，因為這樣就能夠達到很極端的配置了。
    ex : 假設經過 tanh 之後輸出 : [1, -1, -1] (這已經是最極端的分配的)，乘上 5 之後就變成 [5, -5, -5] 會得到 [0.9999..., 4.5396e-05, 4.5396e-05]

    SAC 也一定要使用 tanh，我試過直接不用 tanh 會出現 bang-bang control 問題。
'''

exps_fixed = ['exp37', 'exp38', 'exp39', 'exp40', 'exp41']
# exps_moving = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
exps_moving = ['exp70']
# seeds = [124, 125, 126, 127, 128]
seeds = [124]
fixed_or_not = [False]
hard_scenario = False
new_mimo_scenario = False
using_tanh = True
nr_oriented_scenario = True

for i in range(len(seeds)):

    for fixed in fixed_or_not:

        if fixed: exps = exps_fixed
        else: exps = exps_moving

        print(f"It's {exps[i]}, seeds {seeds[i]}")
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''設定環境變數'''
        set_seed(seed= seeds[i])
        fixed_UE = fixed  # True if using GANDDQN env, False if LSTM_A2C env
        if fixed_UE: print("\n================================================== GANDDQN_env ==================================================\n")
        else: print("\n================================================== LSTM-A2C_env ==================================================\n")

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''設定 tensorboard'''
        algo_name = 'SAC'
        exp_name = exps[i]
        log_file = 'Logs_github' if fixed_UE == False else 'Logs_fixedUE_env'
        log_path = Path(f"{log_file}/{algo_name}/{exp_name}/tensorboard")
        # generate log writer
        writer = SummaryWriter(log_dir= log_path)

        # 要看 tensorboard 結果，輸入在 terminal 中他會給你一個網址
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/"algo_name"/"exp_name"/tensorboard"
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/SAC/exp5/tensorboard"
        # 程式跑下去之後就可以用另一個 terminal 開啟 tensorboard，接著你任何時候想看進度就去點一下 tensorboard 頁面的重置就好了

        if fixed_UE: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/SAC") / f"{exp_name}"
        else: image_path = Path(f"Temp_Figures/{algo_name}") / f"{exp_name}"
        # 自行偵測資料夾，若不存在就補上，若存在也不報錯
        # parents= True -> 更上層的資料夾一併檢查補上
        # exist_ok= True -> 若已經存在也不會報錯
        image_path.mkdir(parents=True, exist_ok=True)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Moving Average'''
        # np.convolve(data, kernel= np.ones(window_size) / window_size, mode= 'valid')，用 kernel 掃過整個 data (stride = 1)
        # kernel : if window_size = 3, then kernel = [1/3, 1/3, 1/3]. 可以想成是每一個資料所佔的比例
        # mode= 'valid'，不做 padding，只對完整的 window 做 moving average
        def moving_average(data, window_size):
            data = np.array(data)
            return np.convolve(data, np.ones(window_size) / window_size, mode= 'valid')

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''State Preprocessing'''
        # state : np.array, shape (state_dim)
        # ser_cat : list, len = 3
        # return preprocessed state : np.array, shape (state_dim)
        # def state_preprocessing(state):
        #     preproc_state = np.zeros(state.shape)
        #     # if state = [0, 0, 0], then return np.array([0, 0, 0])
        #     if state.sum() == 0: return preproc_state
        #     # if any element in state is not 0, then normalize the state (z-score normalization)
        #     else: 
        #         preproc_state = state.copy()
        #         return (preproc_state - preproc_state.mean()) / preproc_state.std()

        # 改用 max-scaling (讓輸出介於 [0~1])
        # 因為用 z-score 沒辦法體現當前流量的負載是忙碌還是很輕鬆，乍看之下根本環境沒差，但其實有
        # ex:
        # 負載輕鬆 : slice A 需要 1 單位，slice B 需要 9 單位資源 -> Z-score 視角：A 很小 (負值)，B 很大 (正值)。模型決定給 Slice A 10% (1 MHz)，Slice B 90% (9 MHz) -> ok
        # 負載很大 : slice A 需要 100 單位，slice B 需要 900 單位 -> Z-score 只看分佈，這兩個數字經過正規化後，會跟場景 1 幾乎一模一樣！ 模型看到 A 很小 (負值)，B 很大 (正值)。
        #           模型決策：模型回想起場景 1 的成功經驗，再次決定給 Slice A 10% (1 MHz)，Slice B 90% (9 MHz)。
        #           Slice A 這次負載很重，它至少需要 2 MHz 才能活命 (SLA 門檻)，結果你只給它 1 MHz。
        #           結局：Slice A 直接死亡 (SSR=0)。
        # 但觀察後發現本實驗環境中沒有這種情況發生，每個 learning window 的 loading 都差不多。不會有突然暴衝的情況發生。
        # def state_preprocessing(state):
        #     Max_ = 10000000
        #     preproc_state = np.zeros(state.shape)
        #     if state.sum() == 0:  return preproc_state
        #     else: 
        #         preproc_state = state.copy()
        #         preproc_state = preproc_state / Max_
        #     return preproc_state

        # 改用 log-scaling，這應該是最適合這種跨度大的前處理方式
        # state : np.array, shape (state_dim)
        # ser_cat : list, len = 3
        # return preprocessed state : np.array, shape (state_dim)
        def state_preprocessing(state):
            log_state = np.log1p(state)  # 1e^9 -> 9*ln(1) ~ 20.7
            return log_state / 10.0  # 壓到 [0~10] 之間

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Reward Calculation'''
        # cal reward based on utility = \alpha * SE + (\betas * SSRs).sum() after a learning window
        # qoe : SSRs of 3 NS of a complete learning window, np.array, shape (3)
        # qoe_weights : list, len = 3
        # se : average SE of a timeslot of a complete learning window, np.array, shape (1)
        # se_weight : float
        # reward_clipping : clip the reward or not
        # return utility, reward, float (np.array with shape (1))
        # def cal_reward(qoe, se, qoe_weights, se_weight, reward_clipping= False):
        #     utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se
        #     if reward_clipping: 
        #         threshold1 = 6.5
        #         threshold2 = 4.5
        #         if utility >= threshold1: reward = 1
        #         elif utility < threshold1 and utility > threshold2: reward = 0
        #         else: reward = -1   # reward : shape ()
        #     else: reward = utility  # reward : shape (1)
        #     return utility, reward

        # 這種在 fixedUE 中表現跟上一種差不多，但在 movingUE 中表現差於上一種非常多
        # reward : shape (1), utility.shape (1)
        # se : np.int with shape (1), qoe : np.array with shape (3)
        # def cal_reward(qoe, se, qoe_weights, se_weight, reward_clipping= False):
        #     utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]  # shape (1)
        #     if qoe[1] >= 0.98 and qoe[0] >= 0.98:
        #         if qoe[2] >= 0.95:
        #             if se[0] < 280:
        #                 reward = 4
        #             else:
        #                 reward = 4 + (se[0] - 280) * 0.1
        #         else:
        #             reward = (qoe[2] - 0.7) * 10
        #     else:
        #         reward = -5
        #     reward = np.array([reward])

        #     return utility, reward

        # 自創 reward function
        # reward : shape (1), utility.shape (1)
        # se : np.int with shape (1), qoe : np.array with shape (3)
        # def cal_reward(qoe, se, qoe_weights, se_weight, reward_clipping= False):
        #     standard = 0.98  # standard for embb & volte
        #     standard2 = 0.95  # standard for urllc (最高可以 0.96)
        #     utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]  # shape (1)
        #     if qoe[1] >= standard and qoe[0] >= standard:
        #         if qoe[2] >= standard2:
        #             reward = (np.matmul(qoe_weights, qoe.reshape((3, 1))) + (se_weight / 100.0) * se[0])[0] / 10  # 會介於 0~1
        #         else:
        #             reward = (qoe[2] - standard2) - 0.5  # -0.5~-1.45
        #     else:
        #         reward = -1.5  - max(0, standard - qoe[0]) - max(0, standard - qoe[1])
        #     reward = np.array([reward])

        #     return utility, reward

        # 自創 reward function2 -> 就分兩種情況就好，一種是已經滿足所有 SSR，另一種是滿足所有 SSR 了。
        # reward : shape (1), utility.shape (1)
        # se : np.int with shape (1), qoe : np.array with shape (3)
        # def cal_reward(qoe, se, qoe_weights, se_weight, reward_clipping= False):
        #     SLA_threshold = 0.95
        #     utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]  # shape (1)
        #     if ((qoe[2] >= SLA_threshold) and (qoe[1] >= SLA_threshold) and (qoe[0] >= SLA_threshold)):
        #         reward = (np.matmul(qoe_weights, qoe.reshape((3, 1))) + (se_weight / 1.0) * se[0])[0] / 10  # 會介於 0~1
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
            
            return utility, reward

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Setup device'''
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Setup envirnoment & related Parameters'''
        ser_cat = ['volte', 'embb_general', 'urllc']
        qoe_weights = [1, 1, 1]
        se_weight = 0.01
        
        '''total bandwidth'''
        if hard_scenario or nr_oriented_scenario : total_band = 20 * 10**6  # unit : MHz
        elif new_mimo_scenario: total_band = 40 * 10**6
        else: total_band = 10 * 10**6
        '''dl_mimo'''
        if hard_scenario : dl_mimo = 3  # 3
        elif new_mimo_scenario: dl_mimo = 4
        else: dl_mimo = 16
        '''UE_rx_gain'''
        if new_mimo_scenario: rx_gain = 1
        else: rx_gain = 20
            
        if nr_oriented_scenario:  # 5G NR scenario (TR 38.901)
            learning_windows = 200
            RB_band = 360 * 10 ** 3
            chan_mod = '38901_UMi_NLOS'
            carrier_freq = 3.5 * 10 ** 9
        else: # 4G LTE scenario (Original) TR 36.814
            learning_windows = 2000  # 1 learning window (episode) = 2000 timeslots
            RB_band = 180 * 10 ** 3
            chan_mod = '36814'
            carrier_freq = 2 * 10 ** 9

        total_timesteps = 10000
        UE_no = 100 if fixed_UE else 300
        if fixed_UE: env = cellularEnv(
            ser_cat= ser_cat, 
            ser_prob= np.array([6, 6, 1], dtype= np.float32), 
            band_whole = total_band,
            learning_windows= learning_windows, 
            dl_mimo= dl_mimo, 
            rx_gain= rx_gain,
            UE_max_no= UE_no, 
            hard_scenario= hard_scenario,
            new_mimo_scenario= new_mimo_scenario)
        else: env = EnvMove(
            UE_max_no= UE_no, 
            ser_prob= np.array([6, 6, 1], dtype= np.float32), 
            band_whole= total_band,
            learning_windows= learning_windows, 
            dl_mimo= dl_mimo, 
            rx_gain= rx_gain,
            hard_scenario= hard_scenario,
            new_mimo_scenario= new_mimo_scenario,
            speed_each_slice= [3, 4, 9],
            RB_band= RB_band,
            chan_mod= chan_mod,
            carrier_freq= carrier_freq)
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Setup Training Parameters'''
        batch_size = 32
        buffer_size = 10000
        gamma = 0.99  # discount factor
        tau = 0.005  # soft update params
        lr = 3e-4  # learning rate of actor & critic
        alpha_lr = 3e-4  # learning rate of alpha (used in controlling the impact of the Entropy)
        if using_tanh: ACTION_SCALE = 3.0
        else: ACTION_SCALE = 1.0
        state_dim = len(ser_cat)
        action_dim = len(ser_cat)

        Actor = Model.Actor(state_dim= state_dim, action_dim= action_dim).to(device= DEVICE)
        Actor_optim = torch.optim.Adam(Actor.parameters(), lr= lr)
        Critic = Model.DoubleCritic(state_dim= state_dim, action_dim= action_dim).to(device= DEVICE)
        Critic_optim = torch.optim.Adam(Critic.parameters(), lr= lr)
        Target_Critic = Model.DoubleCritic(state_dim= state_dim, action_dim= action_dim).to(device= DEVICE)
        Buffer = ReplayBuffer.ReplayBuffer(capacity= buffer_size, device= DEVICE)
        Sacopt = SACopt.SAC_opt(
            state_dim= state_dim,
            action_dim= action_dim,
            actor= Actor,
            critic= Critic,
            target_critic= Target_Critic,
            actor_optim= Actor_optim,
            critic_optim= Critic_optim,
            gamma= gamma,
            tau= tau,
            device= DEVICE,
            alpha_lr= alpha_lr
        )

        # recording lists
        QoEs = []
        SEs = []
        Utilities = []
        Rewards = []
        Observations = []
        Actor_losses = []
        Critic_losses = []

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Training'''
        env.countReset()  # reset 所有計數器
        if not fixed_UE: env.user_move()  # user move in LSTM-A2C env
        env.activity()  # 所有 UE 開始根據其網路切片產生封包
        # observation_packets : total packets of each NSs, np.array with shape (state_dim)
        # observation_bits : total bits of each NSs, np.array with shape (state_dim)
        observation_packets, observation_bits = env.get_state()  
        state = state_preprocessing(observation_bits)  # np.array with shape (3)
        
        inference_time_ms = 0
        for frame in tqdm(range(1, total_timesteps+1)):
            print(f"\n\n******Episode {frame} :")
            # ACTION_SCALE = min(1.0, 1.0 + (frame / 5000.0) * 2.0)
            if using_tanh: action_logit, inference_time_ms = Sacopt.select_action(state= state)  # tensor (cpu) with shape (3)
            else: action_logit = Sacopt.select_action_no_tanh(state= state)
            action_scaled = action_logit * ACTION_SCALE
            action = F.softmax(action_scaled, dim= 0) * total_band  # tensor (cpu) with shape (3)
            env.band_ser_cat = action.numpy()  # apply action to the environment
            # lower layer
            for _ in range(learning_windows):
                env.scheduling()
                env.provisioning()
                env.activity()
            # qoe : np.array with shape (3)
            # se : np.array with shape (1)
            qoe, se = env.get_reward()
            # utility, reward : np.array with shape (1)
            utility, reward = cal_reward(qoe= qoe, se= se, qoe_weights= qoe_weights, se_weight= se_weight, reward_clipping= False)

            throughput = env.get_throughput()  # 取得這一個 window 的 throughput (bps)
            throughput_mbps = throughput / 1e6
            
            print(f"\ninference time (ms) = {inference_time_ms}")
            writer.add_scalar(tag= 'time/inference_ms', scalar_value= inference_time_ms, global_step= frame)
            writer.add_scalar(tag= 'observationBits/volte', scalar_value= observation_bits[0], global_step= frame)
            writer.add_scalar(tag= 'observationBits/embb_general', scalar_value= observation_bits[1], global_step= frame)
            writer.add_scalar(tag= 'observationBits/urllc', scalar_value= observation_bits[2], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/volte', scalar_value= observation_packets[0], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/embb_general', scalar_value= observation_packets[1], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/urllc', scalar_value= observation_packets[2], global_step= frame)

            observation_packets, observation_bits = env.get_state()  
            next_state = state_preprocessing(observation_bits)
            
            Buffer.store_exp(state= state, action= action_logit, reward= reward[0], next_state= next_state)
            # update Models
            if len(Buffer) > batch_size * 3:
                if using_tanh:
                    critic_loss, actor_loss, alpha_loss, alpha = Sacopt.update(buffer= Buffer, batch_size= batch_size)
                else:
                    critic_loss, actor_loss, alpha_loss, alpha = Sacopt.update_no_tanh(buffer= Buffer, batch_size= batch_size)
                loss = {
                    'actor_loss' : actor_loss,
                    'critic_loss' : critic_loss,
                    'alpha_loss' : alpha_loss
                }
                pprint(loss)
                writer.add_scalar(tag= 'loss/actor_loss', scalar_value= actor_loss, global_step= frame)
                writer.add_scalar(tag= 'loss/critic_loss', scalar_value= critic_loss, global_step= frame)
                writer.add_scalar(tag= 'loss/alpha_loss', scalar_value= alpha_loss, global_step= frame)
            
            # calculate the individual se of each network slices of the current learning window
            # indivifual_se : np.array with shape (3)
            # urllc_perfect, tolerable, fail : packet count categorized by latency for transmitted URLLC traffic of the current learning window, int
            individual_se, urllc_perfect, urllc_tolerable, urllc_fail, idle_frame = env.eval_get_obs()

            # print the outcome of the current learning window
            # print(f"action = {((action[0] / total_band).item()):.3f}, {((action[1] / total_band).item()):.3f}, {((action[2] / total_band).item()):.3f}")
            print(f"qoe = {qoe}, se = {float(se[0]):.3f}, reward = {float(reward[0]):.3f}, utility = {float(utility[0]):.3f}, throughput = {throughput_mbps: .3f} Mbps")

            # Record the values of the current learning window
            QoEs.append(qoe.tolist())  # qoe.tolist() -> [qoe1, qoe2, qoe3]
            SEs.append(se.tolist()[0])  # se.tolist() -> [se]
            Rewards.append(reward.item())  
            Utilities.append(utility.item())
            
            # record training arguments
            writer.add_scalar(tag= 'qoe/volte', scalar_value= qoe[0], global_step= frame)
            writer.add_scalar(tag= 'qoe/embb_general', scalar_value= qoe[1], global_step= frame)
            writer.add_scalar(tag= 'qoe/urllc', scalar_value= qoe[2], global_step= frame)
            writer.add_scalar(tag= 'se', scalar_value= se[0], global_step= frame)
            writer.add_scalar(tag= 'reward', scalar_value= reward[0], global_step= frame)
            writer.add_scalar(tag= 'utility', scalar_value= utility[0], global_step= frame)
            writer.add_scalar(tag= 'individual_se/volte', scalar_value= individual_se[0], global_step= frame)
            writer.add_scalar(tag= 'individual_se/embb_general', scalar_value= individual_se[1], global_step= frame)
            writer.add_scalar(tag= 'individual_se/urllc', scalar_value= individual_se[2], global_step= frame)
            writer.add_scalar(tag= 'pending_packets/volte', scalar_value= env.pending_packets[0], global_step= frame)  # 每一個 window 分完後各網路切片還剩下多少待傳的 buffer
            writer.add_scalar(tag= 'pending_packets/embb_general', scalar_value= env.pending_packets[1], global_step= frame)
            writer.add_scalar(tag= 'pending_packets/urllc', scalar_value= env.pending_packets[2], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/perfect', scalar_value= urllc_perfect, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/tolerable', scalar_value= urllc_tolerable, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail', scalar_value= urllc_fail, global_step= frame)
            writer.add_scalar(tag= 'action/volte', scalar_value= action.numpy()[0], global_step= frame)  # 分配比例
            writer.add_scalar(tag= 'action/embb_general', scalar_value= action.numpy()[1], global_step= frame)
            writer.add_scalar(tag= 'action/urllc', scalar_value= action.numpy()[2], global_step= frame)
            writer.add_scalar(tag= 'throughput', scalar_value= throughput_mbps, global_step= frame)
            
            env.countReset()  # reset 所有計數器
            if not fixed_UE: env.user_move()  # user move in LSTM-A2C env

            state = next_state
        
        # if fixed_UE:
        #     torch.save(Actor.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/fixed_UE/6_algos/SAC/actor_weights.pth')
        #     torch.save(Critic.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/fixed_UE/6_algos/SAC/critic_weights.pth')
        # else:
        #     torch.save(Actor.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/movingUE/6_algos/SAC/actor_weights.pth')
        #     torch.save(Critic.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/movingUE/6_algos/SAC/critic_weights.pth')
        # print("Complete")
        

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Generate Outcome Figures'''

        # QoE_volte, embb, urllc 為長度 10000 的 list
        qoe_volte = [v for (v, e, u) in QoEs]
        qoe_embb = [e for (v, e, u) in QoEs]
        qoe_urllc = [u for (v, e, u) in QoEs]

        # utilities_ 長度 10000 的 list
        Utilities = Utilities[1: ]  # 去除第一筆
        Utilities_ = [u for u in Utilities]

        # use moving average to smooth the curve
        ma_qoe_volte = moving_average(qoe_volte, window_size = 200)
        ma_qoe_embb = moving_average(qoe_embb, window_size = 200)
        ma_qoe_urllc = moving_average(qoe_urllc, window_size = 200)
        ma_SE = moving_average(SEs, window_size = 200)
        ma_utility = moving_average(Utilities_, window_size = 200)
        # ma_actor_loss = moving_average(Actor_losses, window_size= 200)
        # ma_critic_loss = moving_average(Critic_losses, window_size= 200)

        # qoe figure (figure(3))
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

        # se figure (figure(4))
        plt.figure(4)
        plt.clf()
        plt.title('SE')
        plt.xlabel('Episode')
        plt.ylabel('bits/Hz')
        plt.plot(ma_SE)
        plt.savefig(image_path / f"SE.png")

        # utility figure (figure(5))
        plt.figure(5)
        plt.clf()
        plt.title('Utility')
        plt.xlabel("Episode")
        plt.ylabel("utility")
        plt.plot(ma_utility)
        plt.savefig(image_path / f"Utility.png")

        # loss figure (figure(6))
        # plt.figure(6)
        # plt.clf()
        # plt.title('Loss')
        # plt.xlabel("Episode")
        # plt.ylabel("loss")
        # plt.plot(Actor_losses, label= 'actor_loss')
        # plt.plot(Critic_losses, label= 'critic_loss')
        # plt.legend()
        # plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/D2AC/Losses.png")

        print("Graph Saved")




