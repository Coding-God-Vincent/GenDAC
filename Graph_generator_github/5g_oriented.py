import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Exponential Moving Average, bigger the weight (0~1) smoother the line
def ema(values, weight= 0.9):
    values = np.asarray(values, dtype=float)
    smoothed = np.zeros_like(values)  # 創建一個跟 values 一樣大的 np.zeros
    last = values[0]
    for i, v in enumerate(values):
        last = weight * last + (1 - weight) * v
        smoothed[i] = last
    return smoothed


# used in generating the outcome figure
# np.convolve(data, kernel= np.ones(window_size) / window_size, mode= 'valid')，用 kernel 掃過整個 data (stride = 1)
# kernel : if window_size = 3, then kernel = [1/3, 1/3, 1/3]. 可以想成是每一個資料所佔的比例
# mode= 'valid'，不做 padding，只對完整的 window 做 moving average
def moving_average(data, window_size):
    data = np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode= 'valid')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''Hyperparameter'''

window_size = 200
steps = np.arange(9801)
alpha = 0.1  # 透明度
weight = 0.97  # 平滑程度 (除 qoe 其他是 0.9)
color_d2ac = 'tab:red'
color_ganddqn = 'tab:green'
color_hard_slicing = 'tab:blue'
color_lstm_a2c = 'tab:orange'
color_ppo = 'tab:pink'
color_sac = 'tab:cyan'

Figure = "Test_Figures"

algo_name1 = "GenDAC_5g"
algo_name2 = "GANDDQN_5g"
algo_name3 = "LSTM_A2C_5g"
algo_name4 = "Hard_Slicing_5g"
algo_name5 = "PPO_5g"
algo_name6 = "SAC_5g"

algo_names = [algo_name1, algo_name2, algo_name3, algo_name4, algo_name5, algo_name6]
labels = ["GenDAC", "GAN-DDQN", "LSTM-A2C", "Hard Slicing", "PPO", "SAC"]
colors = [color_d2ac, color_ganddqn, color_lstm_a2c, color_hard_slicing, color_ppo, color_sac]
seeds = [124]
zorders = [11, 5, 12, 10, 9, 8]
linewidths = [1.8, 1, 1, 1, 1, 1]
# alphas = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
alphas = [0.12, 0.08, 0.08, 0.08, 0.08, 0.08]

image_path = Path(f"{Figure}/5g_oriented")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''Utility'''


means_across_algos = []
stds_across_algos = []


# calculate std & mean of each algo
for i, algo_name in enumerate(algo_names):
    values = []
    for j in range(len(seeds)):
        # set csv path
        csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
        values.append(moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "utility.csv")['Value'], window_size= window_size))  # (5, 10000)
    values = np.array(values)
    means_across_algos.append(np.mean(values, axis= 0))  # (6, 10000)
    stds_across_algos.append(np.std(values, axis= 0))  # (6, 10000)

# generate figure
plt.figure(0)
plt.xlabel("Decision Window")
plt.ylabel("System Utility")
plt.title("Training Utility")

for i in range(len(algo_names)):
    upper = means_across_algos[i] + stds_across_algos[i]
    lower = means_across_algos[i] - stds_across_algos[i]
    plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], zorder= zorders[i], linewidth= 1.4, alpha= 1)
    # plt.plot(upper, linewidth= 1.0, color= colors[i], alpha= 0.5)
    # plt.plot(lower, linewidth= 1.0, color= colors[i], alpha= 0.5)
    plt.fill_between(
        steps,
        upper,
        lower,
        color= colors[i],
        # alpha= alphas[i]
        alpha = 0.15
        # zorder = zorders[i]
)
    
plt.legend(
    fontsize= 'medium',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
    loc= 'lower right'
).set_zorder(10)

plt.savefig(image_path / f"Utility_movingUE.svg")


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''SE'''

means_across_algos = []
stds_across_algos = []

# calculate std & mean of each algo
for i, algo_name in enumerate(algo_names):
    values = []
    for j in range(len(seeds)):
        # set csv path
        csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
        values.append(moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "se.csv")['Value'], window_size= 200))  # (5, 10000)
    values = np.array(values)
    means_across_algos.append(np.mean(values, axis= 0))  # (6, 10000)
    stds_across_algos.append(np.std(values, axis= 0))  # (6, 10000)

# generate figure
plt.figure(0)
plt.clf()
plt.xlabel("Decision Window")
plt.ylabel("SE (bps/Hz)")
plt.title("SE")
# plt.ylim(30, 400)

for i in range(len(algo_names)):
    upper = means_across_algos[i] + stds_across_algos[i]
    lower = means_across_algos[i] - stds_across_algos[i]
    plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], zorder= zorders[i], linewidth= 1.4, alpha= 1)
    # plt.plot(upper, linewidth= 1.0, color= colors[i], alpha= 0.5)
    # plt.plot(lower, linewidth= 1.0, color= colors[i], alpha= 0.5)
    plt.fill_between(
        steps,
        upper,
        lower,
        color= colors[i],
        alpha= alphas[i]
        # zorder = zorders[i]
)
    
plt.legend(
    fontsize= 'medium',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
    loc= 'lower right'
).set_zorder(10)

plt.savefig(image_path / f"SE_movingUE.svg")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''QoE'''

# Hyperparameters
qoes = ['qoe_embb_general', 'qoe_urllc', 'qoe_volte']  # qoe_embb_general, qoe_urllc, qoe_volte

for current_qoe in qoes:
    #-----------------------------------------------------------------------------------------------------------------------------------------------#
    if current_qoe == 'qoe_embb_general': title = ' SSR of eMBB service'
    elif current_qoe == 'qoe_urllc' : title = 'SSR of URLLC service'
    else: title = "SSR of VoLTE service"

    
    means_across_algos = []
    stds_across_algos = []

    # calculate std & mean of each algo
    for i, algo_name in enumerate(algo_names):
        values = []
        for j in range(len(seeds)):
            # set csv path
            csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
            values.append(moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / f"{current_qoe}.csv")['Value'], window_size= window_size))  # (5, 10000)
        values = np.array(values)
        means_across_algos.append(np.mean(values, axis= 0))  # (6, 10000)
        stds_across_algos.append(np.std(values, axis= 0))  # (6, 10000)

    # generate figure
    plt.figure()
    plt.clf()
    plt.xlabel("Decision Window")
    plt.ylabel("SSR")
    plt.title(title)
    plt.ylim(0.0, 1.01)
    # plt.axhline(0.95, linestyle="--", alpha= 0.85, linewidth= 1, color= "k", label= "Effective Gradient Threshold")  # 用虛線標示出有效梯度的界線

    for i in range(len(algo_names)):
        # upper = means_across_algos[i] + stds_across_algos[i]
        # lower = means_across_algos[i] - stds_across_algos[i]
        plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], zorder= zorders[i], linewidth= linewidths[i], alpha= 1)
        # plt.plot(upper, linewidth= 1.0, color= colors[i], alpha= 0.5)
        # plt.plot(lower, linewidth= 1.0, color= colors[i], alpha= 0.5)
        # plt.fill_between(
        #     steps,
        #     upper,
        #     lower,
        #     color= colors[i],
        #     alpha= alphas[i]
        #     # zorder = zorders[i]
        # )
        
    plt.legend(
        fontsize= 'medium',  # 字體大小
        labelspacing= 0.2,  # 垂直標籤之間的間距
        handletextpad= 0.5,  # 圖示與文字之間的間距
        borderaxespad= 0.5,  # 圖例框與邊框的間距
        # ncol= 1  # 2 : 橫向、1 : 垂直
        loc= 'lower right'
    ).set_zorder(10)

    plt.savefig(image_path / f"{current_qoe}.svg")




