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
from Utils.PPO_utils import RolloutBuffer, PPOopt, Model
from Utils.seed import set_seed
from pprint import pprint

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''設定環境變數'''
set_seed(seed= 123)
fixed_UE = False  # True if using GANDDQN env, False if LSTM_A2C env
if fixed_UE: print("\n================================================== GANDDQN_env ==================================================\n")
else: print("\n================================================== LSTM-A2C_env ==================================================\n")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''設定 tensorboard'''
algo_name = 'PPO'
exp_name = 'exp1'
log_file = 'Logs_movingUE_env' if fixed_UE == False else 'Logs_fixedUE_env'
log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs") /log_file / algo_name / exp_name / 'tensorboard'
# generate log writer
writer = SummaryWriter(log_dir= log_path)

# 要看 tensorboard 結果，輸入在 terminal 中他會給你一個網址
# tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/"algo_name"/"exp_name"/tensorboard"
# tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_movingUE_env/PPO/exp1/tensorboard"
# 程式跑下去之後就可以用另一個 terminal 開啟 tensorboard，接著你任何時候想看進度就去點一下 tensorboard 頁面的重置就好了

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
def state_preprocessing(state):
    Max_ = 10000000
    preproc_state = np.zeros(state.shape)
    if state.sum() == 0:  return preproc_state
    else: 
        preproc_state = state.copy()
        preproc_state = preproc_state / Max_
    return preproc_state

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
def cal_reward(qoe, se, qoe_weights, se_weight, reward_clipping= False):
    utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]  # shape (1)
    if qoe[1] >= 0.98 and qoe[0] >= 0.98:
        if qoe[2] >= 0.95:
            if se[0] < 280:
                reward = 4
            else:
                reward = 4 + (se[0] - 280) * 0.1
        else:
            reward = (qoe[2] - 0.7) * 10
    else:
        reward = -5
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
total_band = 10 * 10**6  # unit : MHz
total_timesteps = 10000
dl_mimo = 64
learning_windows = 2000
UE_no = 100 if fixed_UE else 300
if fixed_UE: env = cellularEnv(ser_cat= ser_cat, learning_windows= learning_windows, dl_mimo= 64, UE_max_no= UE_no) 
else: env = EnvMove(UE_max_no= UE_no, ser_prob= np.array([1, 2, 3], dtype= np.float32), learning_windows= learning_windows, dl_mimo= dl_mimo)

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
ACTION_SCALE = 5  # 因為 PPO 的輸出會經過一個 tanh，為了讓 PPO 做出較極端的策略，故乘上一個 5 之後再進入 Softmax
REWARD_SCALE = 5.0
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
    print(f"\n\n******Episode {frame} :")

    # 動態調整 ACTION_SCALE
    ACTION_SCALE = min(6.0, 2.0 + (frame / 5000.0) * 4.0)

    # action_tanh : torch.tensor with shape (action_dim), no grad
    # log_prob : float
    # value : float
    action_tanh, log_prob, value = Ppoopt.rollout(state= state)
    action_scaled = action_tanh * ACTION_SCALE
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
    next_observation_packets, next_observation_bits = env.get_state()  
    next_state = state_preprocessing(next_observation_bits)
    Buffer.store(
        state= state,
        action= action_tanh,
        log_prob= log_prob,
        reward= reward[0] / REWARD_SCALE,
        done= False,
        value= value
    )
    # update models
    if Buffer.len() >= trajectory_length:
        # 取得 last value
        _, _, last_value = Ppoopt.rollout(state= next_state)
        actor_loss, critic_loss, entropy_loss = Ppoopt.update(buffer= Buffer, last_value= last_value, batch_size= batch_size)
        loss = {
            'actor_loss' : actor_loss,
            'critic_loss' : critic_loss,
            'entropy_loss' : entropy_loss
        }
        # adjust learning rate
        scheduler_actor.step()
        scheduler_critic.step()
        # 更新完畢後清空 buffer
        Buffer.clear()
        pprint(loss)
        writer.add_scalar(tag= 'loss/actor_loss', scalar_value= actor_loss, global_step= frame)
        writer.add_scalar(tag= 'loss/critic_loss', scalar_value= critic_loss, global_step= frame)
        writer.add_scalar(tag= 'loss/entropy_loss', scalar_value= entropy_loss, global_step= frame)
        
    
    # print the outcome of the current learning window
    print(f"qoe = {qoe}, se = {float(se[0]):.3f}, reward = {float(reward[0]):.3f}, utility = {float(utility[0]):.3f}")
    
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

    env.countReset()  # reset 所有計數器
    if not fixed_UE: env.user_move()  # user move in LSTM-A2C env
    env.activity()  # 所有 UE 開始根據其網路切片產生封包

    state = next_state

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
plt.legend(["VoLTE", "Video", "URLLC"])
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/PPO/exp1/QoE.png")

# se figure (figure(4))
plt.figure(4)
plt.clf()
plt.title('SE')
plt.xlabel('Episode')
plt.ylabel('bits/Hz')
plt.plot(ma_SE)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/PPO/exp1/SE.png")

# utility figure (figure(5))
plt.figure(5)
plt.clf()
plt.title('Utility')
plt.xlabel("Episode")
plt.ylabel("utility")
plt.plot(ma_utility)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/PPO/exp1/Utility.png")

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
