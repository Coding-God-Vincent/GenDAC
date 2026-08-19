from Utils.LSTM_A2C_utils import utils
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Env.env_fixedUE import cellularEnv
from Env.env_movingUE import EnvMove
from Utils.LSTM_A2C_utils import utils
from Utils.seed import set_seed
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch
import math

'''原論文的 LSTM-A2C 程式碼適用舊版的 Tensorflow 寫，很多版本不相容的問題，因此用 Pytorch 重現。
以結果上來看是有順利收斂，但效能比原論文上的好。除了更快收斂之外各項數值也更高。
收斂的部分 Gemini 說是因為模型初始化的方式不同。舊版 tf 是用 Xavier Uniform，Pytorch 則是用 Kaming Uniform。
'''

# seeds = [124, 125, 126, 127, 128]
seeds = [127, 128]
exps_fixed = ['exp37', 'exp38', 'exp39', 'exp40', 'exp41']
# exps_moving = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
exps_moving = ['exp56', 'exp57']
fixed_or_not = [False]
hard_scenario = False
new_mimo_scenario = False
nr_oriented_scenario = True

for fixed in fixed_or_not:

    if fixed: 
        exps = exps_fixed
    else: 
        exps = exps_moving

    for i in range(len(seeds)):
    
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''環境參數'''
        set_seed(seed= seeds[i])
        fixed_UE = fixed  # True if using GANDDQN env, False if LSTM_A2C env
        if fixed_UE: print("\n================================================== fixed_UE_env ==================================================\n")
        else: print("\n================================================== Moving_UE_env ==================================================\n")

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''設定圖片 / log 路徑'''
        algo_name = 'LSTM_A2C'
        exp_name = exps[i]
        log_file = 'Logs_movingUE_env' if fixed_UE == False else 'Logs_fixedUE_env'
        log_path = Path(f"Logs/{log_file}/{algo_name}/{exp_name}/tensorboard")
        # generate log writer
        writer = SummaryWriter(log_dir= log_path)

        # 要看 tensorboard 結果，輸入在 terminal 中他會給你一個網址
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/"algo_name"/"exp_name"/tensorboard"
        # tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_movingUE_env/LSTM_A2C/exp5/tensorboard"
        # 程式跑下去之後就可以用另一個 terminal 開啟 tensorboard，接著你任何時候想看進度就去點一下 tensorboard 頁面的重置就好了

        if fixed_UE: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/LSTM_A2C") / f"{exp_name}"
        else: image_path = Path(f"Temp_Figures/{algo_name}") / f"{exp_name}"
        # 自行偵測資料夾，若不存在就補上，若存在也不報錯
        # parents= True -> 更上層的資料夾一併檢查補上
        # exist_ok= True -> 若已經存在也不會報錯
        image_path.mkdir(parents=True, exist_ok=True)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''LSTM-A2C Model & 相關函式'''
        class LSTM_A2C(nn.Module):

            '''Part1 : LSTM-A2C Model'''
            # n_actions : no. of actions space 
            # n_states : observation size (len(ser_cat) = 3) (x_t in LSTM)
            # cell_size : Size of the Cell state (C in LSTM)
            def __init__(
                self,
                n_actions,  # dim of action
                n_state,  # dim of state
                lr_actor= 0.001, 
                lr_critic= 0.1, 
                entropy_beta= 0.001
            ):
                super().__init__()
                self.n_actions = n_actions
                self.n_state = n_state
                self.entropy_beta = entropy_beta
                self.lstm_cell_size = 64  # size of hidden state (h_t) & cell state (C_t) (will be both determined by hidden_size in nn.LSTM)

                '''Network Structure'''
                '''LSTM Layer (Batch_first= True)
                    * def: sequence_length : 一次考慮的時間步, D: LSTM_layers
                    input : shape (batch_size, sequence_length, n_state)
                    h0, c0 : shape (1, batch_size, hidden_size)
                    ex : output, (hn, cn) = nn.LSTM(input, (h0, c0))
                    output : h_t for every t, shape (batch_size, sequence_length, D*hidden_size)
                    h0, c0 : shape (1, batch_size, hidden_size)
                '''
                # input_size : (x_t), which is the n_state in GANDDQN algo. [d1, d2, d3]
                # hidden_size : size of cell state (C_t) and hidden state (h_t)
                # batch_first : let batch_size be in the 0th dimension
                self.lstm_layer = nn.LSTM(input_size= self.n_state, hidden_size= self.lstm_cell_size, batch_first= True)

                '''Critic Layer'''
                # input : hn by lstm_layer with shape (1, batch_size, hidden_size) (will reshape to (batch_size, hidden_size) before input)
                # input_shape (batch_size, hidden_size)
                # output_shape (batch_size, 1)
                self.critic = nn.Sequential(
                    nn.Linear(in_features= self.lstm_cell_size, out_features= 32),
                    nn.Tanh(),
                    nn.Linear(in_features= 32, out_features= 1)
                )

                '''Actor Layer'''
                # input : same as Critic
                # input_shape (batch_size, hidden_size)
                # output_shape (batch_size, n_actions)
                self.actor = nn.Sequential(
                    nn.Linear(in_features= self.lstm_cell_size, out_features= 32),
                    nn.Tanh(),
                    nn.Linear(in_features= 32, out_features= self.n_actions),  # shape (batch_size, n_actions)
                    nn.Softmax(dim= -1)  # shape (batch_size, n_actions)
                )

                '''Optimizer'''
                self.optimizer_actor = torch.optim.Adam(params= self.actor.parameters(), lr= lr_actor)
                # lstm 通常會跟 critic 一起更新，這是因為 lstm 目的為讓 critic 可以準確預測
                # 若 lstm 跟 actor 一起更新，他就會變成為了讓當前策略最大化，而並非客觀地描述環境狀態
                self.optimizer_critic = torch.optim.Adam(params= list(self.critic.parameters()) + list(self.lstm_layer.parameters()), lr= lr_critic)


            # state : sequence_length 個 state in each data in a batch, shape (batch_size, sequence_length, n_states)
            # 因為 lstm 會對這 sequence_length 個 state 提取特徵
            def forward(self, state):
                # hn : shape (1, batch_size, hidden_size)
                output, (hn, cn) = self.lstm_layer(state)
                # we use hn as input of Actor & Critic
                action_probs = self.actor(hn[0])
                v_values = self.critic(hn[0])
                return action_probs, v_values

            
            '''Part2 : 相關函式'''
            # 依照一個 batch 的 states 決定出該 batch 的 actions
            # state : shape (batch_size, sequence_length, n_state)
            def choose_action(self, state):
                with torch.no_grad():
                    # action_probs : shape (batch_size, n_actions)
                    action_probs, _ = self.forward(state= state)
                dice = Categorical(probs= action_probs)  # 創造一顆骰子，可以根據輸出得多組機率分布來決定出多組動作
                actions = dice.sample()  # shape (batch_size)
                return actions.item()


            # 算一個 batch 的 target_v，即 V(s_(t+1))
            # state : s_t, shape (batch_size, sequence_length, n_state)
            def target_v(self, state):
                with torch.no_grad():
                    _, v_values = self.forward(state= state)
                return v_values

            
            # 用一個 batch 的資料來更新 (batch_size = 1 here)
            # state : st, shape (batch_size, sequence_length, n_state)
            # action : int
            # td_target : r_t+V(s_(t+1))，shape (batch_size)
            def learn(self, state, action, td_target):
                # action_probs : shape (batch_size, n_action)
                # v_values : shape (batch_size, 1)
                action_probs, v_values = self.forward(state= state)  
                v_values = v_values.squeeze(dim= 1)  # shape (batch_size)
                '''Critic Loss'''
                critic_loss = F.mse_loss(v_values, td_target.detach())
                '''Actor Loss'''
                dice = Categorical(probs= action_probs)
                entropy = dice.entropy()  # entropy of action_probs
                log_action_probs = dice.log_prob(action)
                actor_loss = - ((td_target.detach() - v_values.detach()) * log_action_probs + self.entropy_beta * entropy)
                total_loss = actor_loss + critic_loss
                '''update Models'''
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                total_loss.backward()
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                return total_loss.item()


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # np.convolve(data, kernel= np.ones(window_size) / window_size, mode= 'valid')，用 kernel 掃過整個 data (stride = 1)
        # kernel : if window_size = 3, then kernel = [1/3, 1/3, 1/3]. 可以想成是每一個資料所佔的比例
        # mode= 'valid'，不做 padding，只對完整的 window 做 moving average
        def moving_average(data, window_size):
            data = np.array(data)
            return np.convolve(data, np.ones(window_size) / window_size, mode= 'valid')
            

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # 改用 log-scaling，這應該是最適合這種跨度大的前處理方式
        # state : np.array, shape (state_dim)
        # ser_cat : list, len = 3
        # return preprocessed state : np.array, shape (state_dim)
        def state_preprocessing(state):
            log_state = np.log1p(state)  # 1e^9 -> 9*ln(1) ~ 20.7
            return log_state / 10.0  # 壓到 [0~10] 之間
        
        # 原始 state preprocessing
        # def state_preprocessing(pkt_nums):
        #     mean = np.array([218.8, 5338, 293])
        #     std = np.array([51, 847, 42.5])
        #     state = (pkt_nums - mean) / std
        #     return state

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

        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        '''創建環境並設定相關參數'''
        ser_cat = ['volte', 'embb_general', 'urllc']
        '''total bandwidth'''
        if hard_scenario or nr_oriented_scenario: total_band = 20  # 20MHz (original 10 MHz)
        elif new_mimo_scenario: total_band = 40
        else: total_band = 10
        
        '''dl_mimo'''
        if hard_scenario: dl_mimo = 3  # 原本是 64
        elif new_mimo_scenario: dl_mimo = 4
        else: dl_mimo = 16

        '''UE_rx_gain'''
        if new_mimo_scenario: rx_gain = 1
        else: rx_gain = 20

        if nr_oriented_scenario: 
            learning_windows = 200
            RB_band = 360 * 10 ** 3
        else:
            learning_windows = 2000  # 1 learning window (episode) = 2000 timeslots
            RB_band = 180 * 10 ** 3

        band_per = 0.2  # Granularitiy (unit : MHz)
        total_timesteps = 10000
        
        UE_no = 100 if fixed_UE else 300  # 原本 LSTM-A2C 那邊設 1200 應該是真的沒有 buffer reset，因為 1200 的話要跑超久。這邊為了加速我把人數訂為跟 GANDDQN 那邊一樣 100 人
        if fixed_UE: env = cellularEnv(
            ser_cat= ser_cat, 
            ser_prob= np.array([6, 6, 1], dtype= np.float32), 
            band_whole= total_band * 10 ** 6,
            learning_windows= learning_windows, 
            dl_mimo= dl_mimo, 
            rx_gain= rx_gain,
            UE_max_no= UE_no, 
            hard_scenario= hard_scenario,
            new_mimo_scenario= new_mimo_scenario)
        else: env = EnvMove(
            UE_max_no= UE_no, 
            ser_prob= np.array([6, 6, 1], dtype= np.float32), 
            band_whole= total_band * 10 ** 6,
            learning_windows= learning_windows, 
            dl_mimo= dl_mimo, 
            rx_gain= rx_gain,
            hard_scenario= hard_scenario,
            new_mimo_scenario= new_mimo_scenario,
            speed_each_slice= [3, 4, 9],
            RB_band= RB_band)

        '''GPU'''
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        '''Recording list'''
        QoEs = []
        SEs = []
        Utilities = []
        Rewards = []
        Observations = []
        Actor_losses = []
        Critic_losses = []

        '''Training Parameters'''
        lr_actor = 0.002
        lr_critic = 0.01
        gamma = 0  
        entropy_beta = 0.001
        LSTM_LEN = 10
        # 內含 1128 種組合 (每種組合也都用 list 存)
        action_space = utils.action_space(total= int(total_band // band_per), ser_num= len(ser_cat))  * band_per * 10**6
        # print(len(action_space))  # 1128
        Model = LSTM_A2C(
            n_actions= len(action_space),
            n_state= len(ser_cat),
            lr_actor= lr_actor,
            lr_critic= lr_critic,
            entropy_beta= entropy_beta
        ).to(device= DEVICE)
        lstm_buffer = []

        '''Training'''
        # Prefill : 為了使用 LSTM，先產生 LSTM_LEN 筆資料
        for i in range(LSTM_LEN):
            env.countReset()  # reset 所有計數器
            if not fixed_UE: env.user_move()
            env.activity()  # 所有 UE 開始根據其所屬的網路切片開始產生封包
            action = np.random.choice(len(action_space))
            env.band_ser_cat = action_space[action]

            for i_subframe in range(learning_windows):
                env.scheduling()
                env.provisioning()
                env.activity()
            
            observation_packets, observation_bits = env.get_state()
            observe = state_preprocessing(observation_bits)
            # print(observation_packets, observe)
            lstm_buffer.append(observe)

        # Training
        for frame in tqdm(range(1, total_timesteps+1)):
            print(f"\n\n******Episode {frame} :")
            env.countReset()  # reset 所有計數器
            if not fixed_UE: env.user_move()
            state = np.vstack(lstm_buffer)  # lstm_buffer : np.array with shape (sequence_length (LSTM_LEN), n_state)
            state = torch.from_numpy(state).to(device= DEVICE, dtype= torch.float32).unsqueeze(dim= 0)  # shape (1, sequence_length, n_state)
            action = Model.choose_action(state= state)  # int
            env.band_ser_cat = action_space[action]
            # lower level
            for i_subframe in range(learning_windows):
                env.scheduling()
                env.provisioning()
                env.activity()

            observation_packets, observation_bits = env.get_state()
            observe = state_preprocessing(observation_bits)
            lstm_buffer.pop(0)
            lstm_buffer.append(observe)
            next_state = np.vstack(lstm_buffer)  # lstm_buffer : np.array with shape (sequence_length, n_state)
            next_state = torch.from_numpy(next_state).to(device= DEVICE, dtype= torch.float32).unsqueeze(dim= 0)  # shape (1, sequence_length, n_state)
            qoe, se = env.get_reward()  # se : np.int with shape (1), qoe : np.array with shape (3)
            # utility, reward = utils.calc__reward(qoe= qoe, se= se)  # utility, reward : np.array with shape (1)
            utility, reward, _, _ = cal_reward(qoe= qoe, se= se, qoe_weights= [1, 1, 1], se_weight= 0.01)

            throughput = env.get_throughput()  # 取得這一個 window 的 throughput (bps)
            throughput_mbps = throughput / 1e6
            
            v_values2 = Model.target_v(state= next_state).squeeze(dim= 1)  # (batch_size)
            td_target = reward[0] + gamma * v_values2
            loss = Model.learn(state= state, action= torch.tensor(action, dtype= torch.long, device= DEVICE), td_target= td_target)

            # calculate the individual se of each network slices of the current learning window
            # indivifual_se : np.array with shape (3)
            # urllc_perfect, tolerable, fail : packet count categorized by latency for transmitted URLLC traffic of the current learning window, int
            individual_se, urllc_perfect, urllc_tolerable, urllc_fail, idle_frame = env.eval_get_obs()

            # print the outcome of the current learning window
            print(f"qoe = {qoe}, se = {float(se[0]):.3f}, reward = {float(reward[0]):.3f}, utility = {float(utility[0]):.3f}, loss = {loss:.3f}, throughput = {throughput_mbps: .3f} Mbps")

            QoEs.append(qoe.tolist())  # qoe.tolist() -> [qoe1, qoe2, qoe3]
            SEs.append(se.tolist()[0])  # se.tolist() -> [se]
            Rewards.append(reward[0])
            Utilities.append(utility.item())

            '''record on tensorboard'''
            writer.add_scalar(tag= 'qoe/volte', scalar_value= qoe[0], global_step= frame)
            writer.add_scalar(tag= 'qoe/embb_general', scalar_value= qoe[1], global_step= frame)
            writer.add_scalar(tag= 'qoe/urllc', scalar_value= qoe[2], global_step= frame)
            writer.add_scalar(tag= 'se', scalar_value= se[0], global_step= frame)
            writer.add_scalar(tag= 'reward', scalar_value= reward, global_step= frame)
            writer.add_scalar(tag= 'utility', scalar_value= utility, global_step= frame)
            writer.add_scalar(tag= 'individual_se/volte', scalar_value= individual_se[0], global_step= frame)
            writer.add_scalar(tag= 'individual_se/embb_general', scalar_value= individual_se[1], global_step= frame)
            writer.add_scalar(tag= 'individual_se/urllc', scalar_value= individual_se[2], global_step= frame)
            writer.add_scalar(tag= 'pending_packets/volte', scalar_value= env.pending_packets[0], global_step= frame)  # 每一個 window 分完後各網路切片還剩下多少待傳的 buffer
            writer.add_scalar(tag= 'pending_packets/embb_general', scalar_value= env.pending_packets[1], global_step= frame)
            writer.add_scalar(tag= 'pending_packets/urllc', scalar_value= env.pending_packets[2], global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/perfect', scalar_value= urllc_perfect, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/tolerable', scalar_value= urllc_tolerable, global_step= frame)
            writer.add_scalar(tag= 'urllc_packets/fail', scalar_value= urllc_fail, global_step= frame)
            writer.add_scalar(tag= 'action/volte', scalar_value= action_space[action][0], global_step= frame)  # 分配比例
            writer.add_scalar(tag= 'action/embb_general', scalar_value= action_space[action][1], global_step= frame)
            writer.add_scalar(tag= 'action/urllc', scalar_value= action_space[action][2], global_step= frame)
            writer.add_scalar(tag= 'observationBits/volte', scalar_value= observation_bits[0], global_step= frame)
            writer.add_scalar(tag= 'observationBits/embb_general', scalar_value= observation_bits[1], global_step= frame)
            writer.add_scalar(tag= 'observationBits/urllc', scalar_value= observation_bits[2], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/volte', scalar_value= observation_packets[0], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/embb_general', scalar_value= observation_packets[1], global_step= frame)
            writer.add_scalar(tag= 'observationPackets/urllc', scalar_value= observation_packets[2], global_step= frame)
            writer.add_scalar(tag= 'throughput', scalar_value= throughput_mbps, global_step= frame)

        # if fixed_UE == True: torch.save(Model.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/fixed_UE/6_algos/LSTM_A2C/lstm_a2c_weights.pth')
        # else: torch.save(Model.state_dict(), '/home/super_trumpet/NCKU/Paper/My Methodology/Params/movingUE/6_algos/LSTM_A2C/lstm_a2c_weights.pth')
        
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
        plt.figure(0)
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
        plt.figure(1)
        plt.clf()
        plt.title('SE')
        plt.xlabel('Episode')
        plt.ylabel('bits/Hz')
        plt.plot(ma_SE)
        plt.savefig(image_path / f"SE.png")

        # utility figure (figure(5))
        plt.figure(2)
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

