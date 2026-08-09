import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MultipleLocator

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
'''Hyperparameters'''
alpha = 0.1
window = 200
total_step = 9801
color1 = 'tab:red'
color2 = 'tab:green'
color3 = 'tab:blue'
color4 = 'tab:orange'
color5 = 'tab:pink'

Figure = "Test_Figures"

algo1 = "GenDAC_DDPM_1"
algo2 = "GenDAC"
algo3 = "GenDAC_DDPM_5"
algo4 = "GenDAC_DDPM_7"


# 5 steps
# colors = [color1, color2, color3, color4, color5]
# algo_names = [algo1, algo2, algo3, algo4, algo5]
# labels = ["L = 1", "L = 3", "L = 5", "L = 7", "L = 20"]

# 4 steps
colors = [color1, color2, color3, color4, color5]
# algo_names = [algo1, algo2, algo3, algo4]
algo_names = [algo1, algo2, algo3, algo4]
# algo_names = [algo1, algo2, algo3, algo4, algo5]
# algo_names = [algo1, algo2, algo3]
labels = ["L = 1", "L = 3", "L = 5", "L = 7", "L = 9"]
zorders = [5, 10, 5, 5, 5]
linewidth = 1.0

seeds = [124, 125, 126, 127, 128]

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''1. Utility Learning Curve'''

steps = np.arange(total_step)
means_across_algos = []
stds_across_algos = []

image_path = Path(f"{Figure}/denoise_step")

for label in labels:
    # calculate median & q1 & q3 of each algo
    for i, algo_name in enumerate(algo_names):
        values = []
        for j in range(len(seeds)):
            # set csv path
            csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
            values.append(moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "utility.csv")['Value'], window_size= window))  # (5, 10000)
            # print(i, len(values[-1]))
        values = np.array(values)
        means_across_algos.append(np.mean(values, axis= 0))  # (2, 10000)
        stds_across_algos.append(np.std(values, axis= 0))  # (2, 10000)

# generate figure
plt.clf()
plt.figure(0)
plt.xlabel("Decision Window")
plt.ylabel("System Utility")
plt.title("Training Utility")

for i in range(len(algo_names)):

    upper = means_across_algos[i] + stds_across_algos[i]
    lower = means_across_algos[i] - stds_across_algos[i]
    plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], linewidth= linewidth, alpha= 1, zorder= zorders[i])
    # plt.plot(upper, linewidth= 0.5, color= colors[i], alpha= 0.3)
    # plt.plot(lower, linewidth= 0.5, color= colors[i], alpha= 0.3)
    
    plt.fill_between(
        steps,
        upper,
        lower,
        color= colors[i],
        alpha= alpha
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

plt.savefig(image_path / f"denoise_step_utility.svg")



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''qoe'''

# Hyperparameters
qoes = ['qoe_embb_general', 'qoe_urllc', 'qoe_volte']  # qoe_embb_general, qoe_urllc, qoe_volte

for current_qoe in qoes:

    if current_qoe == 'qoe_embb_general': title = ' SSR of video service'
    elif current_qoe == 'qoe_urllc' : title = 'SSR of URLLC service'
    else: title = "SSR of VoLTE service"
    
    means_across_algos = []
    stds_across_algos = []

    # calculate median & IQR of each algo
    for i, algo_name in enumerate(algo_names):
        values = []

        for j in range(len(seeds)):
            # set csv path
            csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"

            values.append(
                moving_average(
                    pd.read_csv(csv_path / f"{algo_name}_csv" / f"{current_qoe}.csv")['Value'],
                    window_size= window
                )
            )

        values = np.array(values)  # shape: (num_seeds, 10000)
        means_across_algos.append(np.mean(values, axis= 0))  # (2, 10000)
        stds_across_algos.append(np.std(values, axis= 0))  # (2, 10000)

    # generate figure
    plt.figure()
    plt.clf()
    plt.xlabel("Decision Window")
    plt.ylabel("SSR")
    plt.title(title)
    # plt.ylim(0.9, 1.01)
    plt.ylim(0.0, 1.01)

    for i in range(len(algo_names)):
        upper = means_across_algos[i] + stds_across_algos[i]
        lower = means_across_algos[i] - stds_across_algos[i]
        plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], linewidth= linewidth, alpha= 1, zorder= zorders[i])
        # plt.plot(upper, linewidth= 1.0, color= colors[i], alpha= 0.5)
        # plt.plot(lower, linewidth= 1.0, color= colors[i], alpha= 0.5)
        plt.fill_between(
            steps,
            upper,
            lower,
            color= colors[i],
            alpha= alpha
            # zorder = zorders[i]
        )

    plt.legend(
        fontsize='large',
        labelspacing=0.2,
        handletextpad=0.5,
        borderaxespad=0.5,
        loc='lower right'
    ).set_zorder(10)

    plt.savefig(image_path / f"denoise_step_{current_qoe}.svg")
    




