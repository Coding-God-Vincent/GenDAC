import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from Env.env_fixedUE import cellularEnv
from Env.env_movingUE_dynamic import EnvMove
from Utils.PPO_utils import RolloutBuffer, PPOopt, Model
from Utils.seed import set_seed
from pprint import pprint
import math



fixed = False
exps = ['exp39']
seeds = [124]
using_tanh = False
hard_scenario = False

for i in range(len(seeds)):
    
    
    
        print(f"It's {exps[i]}, seeds {seeds[i]}")
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''設定環境變數'''
        set_seed(seed= seeds[i])
        fixed_UE = fixed  # True if using GANDDQN env, False if LSTM_A2C env
        if fixed_UE: print("\n================================================== GANDDQN_env ==================================================\n")
        else: print("\n================================================== LSTM-A2C_env ==================================================\n")

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''設定 tensorboard'''
        algo_name = 'PPO'
        exp_name = exps[i]
        log_file = 'Logs_movingUE_env' if fixed_UE == False else 'Logs_fixedUE_env'
        log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs") /log_file / algo_name / exp_name / 'tensorboard'
        # generate log writer
        writer = SummaryWriter(log_dir= log_path)

        # 要看 tensorboard 結果，輸入在 terminal 中他會給你一個網址
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/"algo_name"/"exp_name"/tensorboard"
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/PPO/exp4/tensorboard"
        # 程式跑下去之後就可以用另一個 terminal 開啟 tensorboard，接著你任何時候想看進度就去點一下 tensorboard 頁面的重置就好了

        if fixed_UE: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/PPO") / f"{exp_name}"
        else: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/PPO") / f"{exp_name}"
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
        if hard_scenario : total_band = 20 * 10**6  # unit : MHz
        else: total_band = 10 * 10**6
        total_timesteps = 10000
        if hard_scenario : dl_mimo = 3  # 3
        else: dl_mimo = 16
        learning_windows = 2000
        UE_no = 100 if fixed_UE else 300
        if fixed_UE: env = cellularEnv(ser_cat= ser_cat, ser_prob= np.array([6, 6, 1], dtype= np.float32), learning_windows= learning_windows, dl_mimo= dl_mimo, UE_max_no= UE_no, hard_scenario= hard_scenario) 
        else: env = EnvMove(
            UE_max_no= UE_no, 
            ser_prob= np.array([6, 6, 1], dtype= np.float32), 
            learning_windows= learning_windows, 
            dl_mimo= dl_mimo, 
            hard_scenario = hard_scenario,
            profile_shift_interval= 3000,
            profile_schedule= [
                ['volte', 'embb_general', 'urllc'],
                ['urllc', 'volte', 'embb_general'],
                ['embb_general', 'urllc', 'volte'],
                ['volte', 'embb_general', 'urllc'],
            ]
        )

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Setup Training Parameters'''
        trajectory_length = 128  # 因為本環境沒有 terminate state，所以自己訂一個 trajectory length (batch_size 的倍數)
        batch_size = 32  
        epochs = 20  # 對一個 trajectory 更新 20 遍
        gamma = 0.99
        gae_lambda = 0.95  # used in GAE
        clip_epsilon = 0.2  # PPO 修正項的上下限所用到的 epsilon
        entropy_coef = 0.0001  # 鼓勵探索的比重
        lr = 1e-3  
        if using_tanh: ACTION_SCALE = 3.0
        else: ACTION_SCALE = 1.0
        
        state_dim = len(ser_cat)
        action_dim = len(ser_cat)

        Actor = Model.Actor(state_dim= state_dim, action_dim= action_dim).to(device= DEVICE)
        Actor_optim = torch.optim.Adam(Actor.parameters(), lr= lr)
        Critic = Model.Critic(state_dim= state_dim).to(device= DEVICE)
        Critic_optim = torch.optim.Adam(Critic.parameters(), lr= lr)
        # 動態調整學習率
        scheduler_actor = torch.optim.lr_scheduler.LinearLR(Actor_optim, start_factor= 1.0, end_factor= 0.01, total_iters= total_timesteps)
        scheduler_critic = torch.optim.lr_scheduler.LinearLR(Critic_optim, start_factor= 1.0, end_factor= 0.01, total_iters= total_timesteps)
        Buffer = RolloutBuffer.RolloutBuffer(device= DEVICE)
        Ppoopt = PPOopt.PPOopt(
            actor= Actor,
            critic= Critic,
            actor_optim= Actor_optim,
            critic_optim= Critic_optim,
            device= DEVICE,
            gamma= gamma,
            gae_lambda= gae_lambda,
            clip_epsilon= clip_epsilon,
            epochs= epochs,
            entropy_coef= entropy_coef
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

        for frame in tqdm(range(1, total_timesteps+1)):

            changed, old_order, new_order = env.update_profile_schedule(frame=frame)

            if changed:
                observation_packets = env.remap_observation_to_current_order(
                    observation= observation_packets,
                    old_order= old_order
                )
                observation_bits = env.remap_observation_to_current_order(
                    observation= observation_bits,
                    old_order= old_order
                )

                # PPO 是 on-policy，profile shift 後不能混舊 trajectory
                Buffer.clear()

                # 很重要：PPO 有額外保存 state 變數，所以 remap 後要重新算 state
                state = state_preprocessing(observation_bits)

                print(f"******[Dynamic Profile Shift] frame={frame}, {old_order} -> {new_order}******")
                print("******[Rollout Buffer Reset]******")

                writer.add_scalar(
                    tag='dynamic/profile_phase',
                    scalar_value=env.current_profile_phase,
                    global_step=frame
                )
                writer.add_text(
                    tag='dynamic/profile_order',
                    text_string=str(env.ser_cat),
                    global_step=frame
                )


            print(f"\n\n******Episode {frame} :")

            # 動態調整 ACTION_SCALE
            # ACTION_SCALE = min(1.0, 1.0 + (frame / 5000.0) * 2.0)


            # 量推論時間
            state_tensor = torch.from_numpy(state).to(
                dtype= torch.float32,
                device= DEVICE
            ).unsqueeze(dim= 0)

            if DEVICE == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            with torch.no_grad():
                if using_tanh:
                    _ = Actor.sample_action(state_tensor)
                else:
                    _ = Actor.sample_action_no_tanh(state_tensor)

            if DEVICE == 'cuda':
                torch.cuda.synchronize()

            inference_time_ms = (time.perf_counter() - t0) * 1000

            writer.add_scalar(
                tag='time/inference_ms',
                scalar_value=inference_time_ms,
                global_step=frame
            )

            # action_tanh : torch.tensor with shape (action_dim), no grad
            # log_prob : float
            # value : float
            if using_tanh: action_logits, log_prob, value = Ppoopt.rollout(state= state)
            else: action_logits, log_prob, value = Ppoopt.rollout_no_tanh(state= state)
            action_scaled = action_logits * ACTION_SCALE
            action = F.softmax(action_scaled, dim= 0) * total_band
            env.band_ser_cat = action.numpy()
            # lower layer
            for _ in range(learning_windows):
                env.scheduling()
                env.provisioning()
                env.activity()
            # qoe : np.array with shape (3)
            # se : np.array with shape (1)
            qoe, se = env.get_reward()
            # reward, utility : np.array with shape (1)
            utility, reward = cal_reward(qoe= qoe, se= se, qoe_weights= qoe_weights, se_weight= se_weight)

            writer.add_scalar(tag= 'observationBits/dim0', scalar_value= observation_bits[0], global_step= frame)
            writer.add_scalar(tag= 'observationBits/dim1', scalar_value= observation_bits[1], global_step= frame)
            writer.add_scalar(tag= 'observationBits/dim2', scalar_value= observation_bits[2], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/dim0', scalar_value= observation_packets[0], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/dim1', scalar_value= observation_packets[1], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/dim2', scalar_value= observation_packets[2], global_step= frame)

            observation_packets, observation_bits = env.get_state()  
            next_state = state_preprocessing(observation_bits)
            Buffer.store(
                state= state,
                action= action_logits,
                log_prob= log_prob,
                reward= reward[0],
                done= False,
                value= value
            )

            # update models
            if Buffer.len() >= trajectory_length:
                # 取得 last value
                if using_tanh:
                    _, _, last_value = Ppoopt.rollout(state= next_state)
                    actor_loss, critic_loss, entropy_loss = Ppoopt.update(buffer= Buffer, last_value= last_value, batch_size= batch_size)
                    loss = {
                        'actor_loss' : actor_loss,
                        'critic_loss' : critic_loss,
                        'entropy_loss' : entropy_loss
                    }
                else:
                    _, _, last_value = Ppoopt.rollout_no_tanh(state= next_state)
                    actor_loss, critic_loss, entropy_loss = Ppoopt.update_no_tanh(buffer= Buffer, last_value= last_value, batch_size= batch_size)
                    loss = {
                        'actor_loss' : actor_loss,
                        'critic_loss' : critic_loss,
                        'entropy_loss' : entropy_loss
                    }
                # adjust learning rate
                # scheduler_actor.step()
                # scheduler_critic.step()
                # 更新完畢後清空 buffer
                Buffer.clear()
                pprint(loss)
                writer.add_scalar(tag= 'loss/actor_loss', scalar_value= actor_loss, global_step= frame)
                writer.add_scalar(tag= 'loss/critic_loss', scalar_value= critic_loss, global_step= frame)
                writer.add_scalar(tag= 'loss/entropy_loss', scalar_value= entropy_loss, global_step= frame)

            # calculate the individual se of each network slices of the current learning window
            # indivifual_se : np.array with shape (3)
            # urllc_perfect, tolerable, fail : packet count categorized by latency for transmitted URLLC traffic of the current learning window, int
            individual_se, urllc_perfect, urllc_tolerable, urllc_fail, idle_frame = env.eval_get_obs()
                
            
            # print the outcome of the current learning window
            print(f"qoe = {qoe}, se = {float(se[0]):.3f}, reward = {float(reward[0]):.3f}, utility = {float(utility[0]):.3f}")
            
            # Record the values of the current learning window
            QoEs.append(qoe.tolist())  # qoe.tolist() -> [qoe1, qoe2, qoe3]
            SEs.append(se.tolist()[0])  # se.tolist() -> [se]
            Rewards.append(reward.item())  
            Utilities.append(utility.item())
            
            # record training arguments
            writer.add_scalar(tag= 'qoe/dim0', scalar_value= qoe[0], global_step= frame)
            writer.add_scalar(tag= 'qoe/dim1', scalar_value= qoe[1], global_step= frame)
            writer.add_scalar(tag= 'qoe/dim2', scalar_value= qoe[2], global_step= frame)
            writer.add_scalar(tag= 'se', scalar_value= se[0], global_step= frame)
            writer.add_scalar(tag= 'reward', scalar_value= reward[0], global_step= frame)
            writer.add_scalar(tag= 'utility', scalar_value= utility[0], global_step= frame)
            writer.add_scalar(tag= 'individual_se/dim0', scalar_value= individual_se[0], global_step= frame)
            writer.add_scalar(tag= 'individual_se/dim1', scalar_value= individual_se[1], global_step= frame)
            writer.add_scalar(tag= 'individual_se/dim2', scalar_value= individual_se[2], global_step= frame)
            writer.add_scalar(tag= 'pending_packets/dim0', scalar_value= env.pending_packets[0], global_step= frame)  # 每一個 window 分完後各網路切片還剩下多少待傳的 buffer
            writer.add_scalar(tag= 'pending_packets/dim1', scalar_value= env.pending_packets[1], global_step= frame)
            writer.add_scalar(tag= 'pending_packets/dim2', scalar_value= env.pending_packets[2], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/perfect', scalar_value= urllc_perfect, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/tolerable', scalar_value= urllc_tolerable, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail', scalar_value= urllc_fail, global_step= frame)
            writer.add_scalar(tag= 'action/dim0', scalar_value= action.numpy()[0], global_step= frame)  # 分配比例
            writer.add_scalar(tag= 'action/dim1', scalar_value= action.numpy()[1], global_step= frame)
            writer.add_scalar(tag= 'action/dim2', scalar_value= action.numpy()[2], global_step= frame)

            env.countReset()  # reset 所有計數器
            if not fixed_UE: env.user_move()  # user move in LSTM-A2C env

            state = next_state
        
        # if fixed_UE:
        #     torch.save(Actor.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/fixed_UE/6_algos/PPO/actor_weights.pth')
        #     torch.save(Critic.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/fixed_UE/6_algos/PPO/critic_weights.pth')
        # else:
        #     torch.save(Actor.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/movingUE/6_algos/PPO/actor_weights.pth')
        #     torch.save(Critic.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/movingUE/6_algos/PPO/critic_weights.pth')

        print("Complete")

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
        plt.legend(["dim0", "dim1", "dim2"])
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
