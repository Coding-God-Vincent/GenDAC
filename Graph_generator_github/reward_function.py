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
color_1 = 'tab:red'
color_2 = 'tab:green'
color_3 = 'tab:blue'
color_4 = 'tab:orange'
# color_ppo = 'tab:pink'
# color_sac = 'tab:cyan'

Figure = "Test_Figures"

algo_name1 = "GenDAC"
algo_name2 = "GenDAC_lstma2c_reward_function"
algo_name3 = "GenDAC_ganddqn_reward_function"
algo_name4 = "GenDAC_weight_sum"


algo_names = [algo_name1, algo_name2, algo_name3, algo_name4]
labels = ["Feasibility-first reward", "LSTM-A2C reward", "GAN-DDQN reward", "weighted sum of SE and SSR"]
colors = [color_1, color_2, color_3, color_4]
seeds = [124, 125, 126, 127, 128]
zorders = [1, 1, 1, 1]
alphas = [0.2, 0.2, 0.2, 0.2]
linewidths = [1, 1, 1, 1]


image_path = Path(f"{Figure}/reward_function")
# Automatically create the output directory if it does not exist
image_path.mkdir(parents=True, exist_ok=True)
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
    fontsize= 'large',  # 字體大小
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
    fontsize= 'large',  # 字體大小
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
        fontsize= 'large',  # 字體大小
        labelspacing= 0.2,  # 垂直標籤之間的間距
        handletextpad= 0.5,  # 圖示與文字之間的間距
        borderaxespad= 0.5,  # 圖例框與邊框的間距
        # ncol= 1  # 2 : 橫向、1 : 垂直
        loc= 'lower right'
    ).set_zorder(10)

    
    plt.savefig(image_path / f"{current_qoe}.svg")


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''avg queue length of each slice'''

# algo_name = 'GenDAC_more_state'
# color_1 = 'tab:red'
# color_2 = 'tab:green'
# color_3 = 'tab:blue'

# volte = moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "avg_queue_length_volte.csv")['Value'], window_size= 200)
# embb = moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "avg_queue_length_embb.csv")['Value'], window_size= 200)
# urllc = moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "avg_queue_length_urllc.csv")['Value'], window_size= 200)

# plt.figure(0)
# plt.clf()
# plt.title('avg. queue length of each slice')
# plt.xlabel('Decision Window')
# plt.ylabel('avg. queue length')

# # volte
# plt.plot(volte, label= 'Volte', color= color_1, linewidth= 1.5)

# # embb
# plt.plot(embb, label= 'eMBB', color= color_2, linewidth= 1.5)

# # urllc
# plt.plot(urllc, label= 'URLLC', color= color_3, linewidth= 1.5)

# plt.legend(
#     fontsize='medium',
#     labelspacing=0.2,
#     handletextpad=0.5,
#     borderaxespad=0.5
#     # loc='lower right'
# ).set_zorder(10)

# plt.savefig(image_path / f"queue_length_{algo_name}.svg")
