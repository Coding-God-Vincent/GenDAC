from Env.env_fixedUE import cellularEnv
from Env.env_movingUE import EnvMove
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Utils.LSTM_A2C_utils.utils import calc__reward
from tqdm.auto import tqdm


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''環境參數'''
fixed_UE = False  # True if using GANDDQN env, False if LSTM_A2C env
if fixed_UE: print("\n================================================== fixed_UE_env ==================================================\n")
else: print("\n================================================== Moving_UE_env ==================================================\n")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''設定圖片 / log 路徑'''
algo_name = 'Hard_Slicing'
exp_name = 'exp1'
log_file = 'Logs_movingUE_env' if fixed_UE == False else 'Logs_fixedUE_env'
log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs") /log_file / algo_name / exp_name / 'tensorboard'
# generate log writer
writer = SummaryWriter(log_dir= log_path)

# 要看 tensorboard 結果，輸入在 terminal 中他會給你一個網址
# tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/"algo_name"/"exp_name"/tensorboard"
# tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/Hard_Slicing/exp1/tensorboard"
# 程式跑下去之後就可以用另一個 terminal 開啟 tensorboard，接著你任何時候想看進度就去點一下 tensorboard 頁面的重置就好了

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# np.convolve(data, kernel= np.ones(window_size) / window_size, mode= 'valid')，用 kernel 掃過整個 data (stride = 1)
# kernel : if window_size = 3, then kernel = [1/3, 1/3, 1/3]. 可以想成是每一個資料所佔的比例
# mode= 'valid'，不做 padding，只對完整的 window 做 moving average
def moving_average(data, window_size):
    data = np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode= 'valid')


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''創建環境並設定相關參數'''
ser_cat = ['volte', 'embb_general', 'urllc']
total_band = 10  # unit : MHz
band_per = 0.2  # Granularitiy (unit : MHz)
total_timesteps = 10000
dl_mimo = 64
learning_windows = 2000
UE_no = 100 if fixed_UE else 1200
if fixed_UE: env = cellularEnv(ser_cat= ser_cat, learning_windows= learning_windows, dl_mimo= 64, UE_max_no= UE_no) 
else: env = EnvMove(UE_max_no= UE_no, ser_prob= np.array([1, 2, 3], dtype= np.float32), learning_windows= learning_windows, dl_mimo= dl_mimo)

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
for frame in tqdm(range(1, total_timesteps+1)):
    print(f"\n\n******Episode {frame} :")
    env.countReset()  # reset 所有計數器
    if not fixed_UE: env.user_move()
    env.activity()  # 所有 UE 開始根據其所屬的網路切片開始產生封包
    action = [total_band/3 * 10**6, total_band/3 * 10**6, total_band/3 * 10**6]  # 平均分給三個網路切片
    env.band_ser_cat = action
    # lower level
    for i_subframe in range(learning_windows):
        env.scheduling()
        env.provisioning()
        env.activity()

    qoe, se = env.get_reward()  # se : np.int with shape (), qoe : np.array with shape (3)
    utility, reward = calc__reward(qoe= qoe, se= se[0])

    # print the outcome of the current learning window
    print(f"qoe = {qoe}, se = {float(se[0]):.3f}, utility = {float(utility):.3f}")

    QoEs.append(qoe.tolist())  # qoe.tolist() -> [qoe1, qoe2, qoe3]
    SEs.append(se.tolist()[0])  # se.tolist() -> [se]
    Utilities.append(utility.item())

    '''record on tensorboard'''
    writer.add_scalar(tag= 'qoe/volte', scalar_value= qoe[0], global_step= frame)
    writer.add_scalar(tag= 'qoe/embb_general', scalar_value= qoe[1], global_step= frame)
    writer.add_scalar(tag= 'qoe/urllc', scalar_value= qoe[2], global_step= frame)
    writer.add_scalar(tag= 'se', scalar_value= se[0], global_step= frame)
    writer.add_scalar(tag= 'utility', scalar_value= utility, global_step= frame)


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
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Hard Slicing/exp1/QoE.png")

# se figure (figure(4))
plt.figure(1)
plt.clf()
plt.title('SE')
plt.xlabel('Episode')
plt.ylabel('bits/Hz')
plt.plot(ma_SE)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Hard_Slicing/exp1/SE.png")

# utility figure (figure(5))
plt.figure(2)
plt.clf()
plt.title('Utility')
plt.xlabel("Episode")
plt.ylabel("utility")
plt.plot(ma_utility)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Hard_Slicing/exp1/Utility.png")