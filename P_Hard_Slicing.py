from Env.env_fixedUE import cellularEnv
from Env.env_movingUE import EnvMove
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Utils.LSTM_A2C_utils.utils import calc__reward
from tqdm.auto import tqdm
from Utils.seed import set_seed
import torch

exps = ['exp6', 'exp7', 'exp8', 'exp9', 'exp10']
seeds = [124, 125, 126, 127, 128]
fixed_or_not = [True, False]
hard_scenario = True

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
for fixed in fixed_or_not:
    for i in range(len(seeds)):
        '''環境參數'''
        set_seed(seed= seeds[i])
        fixed_UE = fixed  # True if using GANDDQN env, False if LSTM_A2C env
        exp_name = exps[i]

        if fixed_UE: print("\n================================================== fixed_UE_env ==================================================\n")
        else: print("\n================================================== Moving_UE_env ==================================================\n")

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''設定圖片 / log 路徑'''
        algo_name = 'P_Hard_Slicing'
        log_file = 'Logs_movingUE_env' if fixed_UE == False else 'Logs_fixedUE_env'
        log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs") /log_file / algo_name / exp_name / 'tensorboard'
        # generate log writer
        writer = SummaryWriter(log_dir= log_path)

        # 要看 tensorboard 結果，輸入在 terminal 中他會給你一個網址
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/"algo_name"/"exp_name"/tensorboard"
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/Hard_Slicing/exp3/tensorboard"
        # 程式跑下去之後就可以用另一個 terminal 開啟 tensorboard，接著你任何時候想看進度就去點一下 tensorboard 頁面的重置就好了

        if fixed_UE: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/P_Hard_Slicing") / exp_name
        else: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/P_Hard_Slicing") / exp_name

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # np.convolve(data, kernel= np.ones(window_size) / window_size, mode= 'valid')，用 kernel 掃過整個 data (stride = 1)
        # kernel : if window_size = 3, then kernel = [1/3, 1/3, 1/3]. 可以想成是每一個資料所佔的比例
        # mode= 'valid'，不做 padding，只對完整的 window 做 moving average
        def moving_average(data, window_size):
            data = np.array(data)
            return np.convolve(data, np.ones(window_size) / window_size, mode= 'valid')


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        def state_preprocessing(state):
            # 作法一: log-scale
            # log_state = np.log1p(state)  # 1e^9 -> 9*ln(1) ~ 20.7
            # return log_state / 10.0  # 壓到 [0~10] 之間

            # 作法二 : 為了保留原始比例，我不用 Log-scale，我直接除以一個同除以一個數字
            return state / 10**7
            


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # 回傳各網路切片所分到的頻寬量 (Hz)
        # state : preprocessed state, np.array, shape (state_dim)
        # total_bandwidth : int, 10* 10**6 (Hz)
        # return logit (np.array with shape (action_dim) , real_action (np.array with shape (action_dim))
        def get_actions(state, total_band):
            state = torch.tensor(state, dtype= torch.float32)  # shape (state_dim)
            proportion = torch.nn.functional.softmax(state, dim= 0).numpy()
            real_action = total_band * proportion
            return real_action


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''創建環境並設定相關參數'''
        ser_cat = ['volte', 'embb_general', 'urllc']
        total_band = 20 * 10**6  # unit : MHz
        band_per = 0.2  # Granularitiy (unit : MHz)
        total_timesteps = 10000
        dl_mimo = 3
        learning_windows = 2000
        UE_no = 100 if fixed_UE else 300
        if fixed_UE: env = cellularEnv(ser_cat= ser_cat, learning_windows= learning_windows, dl_mimo= dl_mimo, UE_max_no= UE_no, hard_scenario= hard_scenario) 
        else: env = EnvMove(UE_max_no= UE_no, ser_prob= np.array([1, 2, 3], dtype= np.float32), learning_windows= learning_windows, dl_mimo= dl_mimo, hard_scenario= hard_scenario)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Recording list'''
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
        if not fixed_UE: env.user_move()
        env.activity()  # 所有 UE 開始根據其所屬的網路切片開始產生封包
        observation_packets, observation_bits = env.get_state()

        for frame in tqdm(range(1, total_timesteps+1)):
            print(f"\n\n******Episode {frame} :")
            state = state_preprocessing(state= observation_bits) 
            action = get_actions(state= state, total_band= total_band)
            env.band_ser_cat = action
            print(f"state = {state}, action= {action}")
            # lower level
            for i_subframe in range(learning_windows):
                env.scheduling()
                env.provisioning()
                env.activity()

            qoe, se = env.get_reward()  # se : np.int with shape (), qoe : np.array with shape (3)
            utility, reward = calc__reward(qoe= qoe, se= se)
            # calculate the individual se of each network slices of the current learning window
            # indivifual_se : np.array with shape (3)
            # urllc_perfect, tolerable, fail : packet count categorized by latency for transmitted URLLC traffic of the current learning window, int
            individual_se, urllc_perfect, urllc_tolerable, urllc_fail, idle_frame = env.eval_get_obs()

            # print the outcome of the current learning window
            print(f"qoe = {qoe}, se = {float(se):.3f}, utility = {float(utility):.3f}")

            QoEs.append(qoe.tolist())  # qoe.tolist() -> [qoe1, qoe2, qoe3]
            SEs.append(se.tolist()[0])  # se.tolist() -> [se]
            Utilities.append(utility.item())

            '''record on tensorboard'''
            writer.add_scalar(tag= 'pending_packets/volte', scalar_value= env.pending_packets[0], global_step= frame)  # 每一個 window 分完後各網路切片還剩下多少待傳的 buffer
            writer.add_scalar(tag= 'pending_packets/embb_general', scalar_value= env.pending_packets[1], global_step= frame)
            writer.add_scalar(tag= 'pending_packets/urllc', scalar_value= env.pending_packets[2], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/perfect', scalar_value= urllc_perfect, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/tolerable', scalar_value= urllc_tolerable, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail', scalar_value= urllc_fail, global_step= frame)
            writer.add_scalar(tag= 'action/volte', scalar_value= action[0], global_step= frame)  # 分配比例
            writer.add_scalar(tag= 'action/embb_general', scalar_value= action[1], global_step= frame)
            writer.add_scalar(tag= 'action/urllc', scalar_value= action[2], global_step= frame)
            writer.add_scalar(tag= 'observationBits/volte', scalar_value= observation_bits[0], global_step= frame)
            writer.add_scalar(tag= 'observationBits/embb_general', scalar_value= observation_bits[1], global_step= frame)
            writer.add_scalar(tag= 'observationBits/urllc', scalar_value= observation_bits[2], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/volte', scalar_value= observation_packets[0], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/embb_general', scalar_value= observation_packets[1], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/urllc', scalar_value= observation_packets[2], global_step= frame)
            writer.add_scalar(tag= 'qoe/volte', scalar_value= qoe[0], global_step= frame)
            writer.add_scalar(tag= 'qoe/embb_general', scalar_value= qoe[1], global_step= frame)
            writer.add_scalar(tag= 'qoe/urllc', scalar_value= qoe[2], global_step= frame)
            writer.add_scalar(tag= 'se', scalar_value= se, global_step= frame)
            writer.add_scalar(tag= 'individual_se/volte', scalar_value= individual_se[0], global_step= frame)
            writer.add_scalar(tag= 'individual_se/embb_general', scalar_value= individual_se[1], global_step= frame)
            writer.add_scalar(tag= 'individual_se/urllc', scalar_value= individual_se[2], global_step= frame)
            writer.add_scalar(tag= 'utility', scalar_value= utility, global_step= frame)

            observation_packets, observation_bits = env.get_state()
            
            env.countReset()  # reset 所有計數器
            if not fixed_UE: env.user_move()

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''Generate Outcome Figures'''
        # QoE_volte, embb, urllc 為長度 10000 的 list
        qoe_volte = [v for (v, e, u) in QoEs]
        qoe_embb = [e for (v, e, u) in QoEs]
        qoe_urllc = [u for (v, e, u) in QoEs]

        # utilities_ 長度 10000 的 list
        # Utilities = Utilities[1: ]  # 去除第一筆
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
        plt.figure(0)
        plt.clf()
        plt.title('QoE')
        plt.xlabel('Episode')
        plt.ylabel('SLA Satisfication Rate')
        plt.plot(ma_qoe_volte)
        plt.plot(ma_qoe_embb)
        plt.plot(ma_qoe_urllc)
        plt.legend(["VoLTE", "Video", "URLLC"])
        plt.savefig(image_path / "QoE.png")

        # se figure (figure(4))
        plt.figure(1)
        plt.clf()
        plt.title('SE')
        plt.xlabel('Episode')
        plt.ylabel('bits/Hz')
        plt.plot(ma_SE)
        plt.savefig(image_path / "SE.png")

        # utility figure (figure(5))
        plt.figure(2)
        plt.clf()
        plt.title('Utility')
        plt.xlabel("Episode")
        plt.ylabel("utility")
        plt.plot(ma_utility)
        plt.savefig(image_path / "Utility.png")