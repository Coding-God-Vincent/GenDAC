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
total_step = 9801

color1 = 'tab:red'
color2 = 'tab:green'
color3 = 'tab:blue'
color4 = 'tab:orange'
color5 = 'tab:cyan'
color6 = 'tab:pink'
color7 = 'tab:gray'

algo1 = "GenDAC_lam_1"
algo2 = "GenDAC_lam_05"
algo3 = "GenDAC_lam_005"
algo4 = "GenDAC_lam_001"
algo5 = "GenDAC_lam_0005"
algo6 = "GenDAC_lam_0001"
algo7 = "GenDAC"
algo8 = "GenDAC_lam_05_0005"
algo9 = "GenDAC_lam_05_00001"



alpha = 0.1
colors = [color1, color2, color3, color4, color5, color6, color7]
zorders = [2, 2, 1, 1, 1, 1, 1]
window_size = 200
seeds = [124, 125, 126, 127, 128]
linewidth = 1
Figure = "Test_Figures"
image_path = Path(f"{Figure}/lambda")


# 7 algos
# algo_names = [algo1, algo2, algo3, algo4, algo5, algo6, algo7]
# labels = [r"$\lambda = 1$", r"$\lambda$ = 0.5", r"$\lambda$ = 0.05", r"$\lambda$ = 0.01", r"$\lambda$ = 0.005", r"$\lambda$ = 0.001", r"$\lambda$ = 0.5_0.001_6000"]

# 7 algos
algo_names = [algo1, algo2, algo6, algo7, algo9]
labels = [r"$\lambda$ = 1", r"$\lambda$ = 0.5", r"$\lambda$ = 0.001", r"$\lambda$ = 0.5_0.001", r"$\lambda$ = 0.5_0.0001"]




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''1. Utility Learning Curve'''

steps = np.arange(total_step)
means_across_algos = []
stds_across_algos = []


for label in labels:
    # calculate median & q1 & q3 of each algo
    for i, algo_name in enumerate(algo_names):
        values = []
        for j in range(len(seeds)):
            # set csv path
            csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
            current_values = pd.read_csv(csv_path / f"{algo_name}_csv" / "utility.csv")['Value']
            if len(current_values) > 10000 : current_values = current_values[0:10000]
            values.append(moving_average(current_values, window_size= window_size))  # (5, 10000)
            # print(f"{len(values[j])}")
        # print(f"{len(values)}, {len(values[1])}")
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
        alpha= alpha,
        zorder = zorders[i]
    )

    
plt.legend(
    fontsize= 'large',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
    loc= 'lower right'
).set_zorder(10)


plt.savefig(image_path / f"lambda_utility.svg")


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''1. SE Learning Curve'''

steps = np.arange(total_step)
means_across_algos = []
stds_across_algos = []


for label in labels:
    # calculate median & q1 & q3 of each algo
    for i, algo_name in enumerate(algo_names):
        values = []
        for j in range(len(seeds)):
            # set csv path
            csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
            current_values = pd.read_csv(csv_path / f"{algo_name}_csv" / "se.csv")['Value']
            if len(current_values) > 10000 : current_values = current_values[0:10000]
            values.append(moving_average(current_values, window_size= window_size))  # (5, 10000)
        values = np.array(values)
        means_across_algos.append(np.mean(values, axis= 0))  # (2, 10000)
        stds_across_algos.append(np.std(values, axis= 0))  # (2, 10000)

# generate figure
plt.clf()
plt.figure(0)
plt.xlabel("Decision Window")
plt.ylabel("SE")
plt.title("SE")

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
        alpha= alpha,
        zorder = zorders[i]
    )

    
plt.legend(
    fontsize= 'large',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
    loc= 'lower right'
).set_zorder(10)

plt.savefig(image_path / f"lambda_se.svg")


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
            csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
            current_values = pd.read_csv(csv_path / f"{algo_name}_csv" / f"{current_qoe}.csv")['Value']
            if len(current_values) > 10000 : current_values = current_values[0:10000]
            values.append(moving_average(current_values, window_size= window_size))  # (5, 10000)
            values.append(
                moving_average(
                    current_values,
                    window_size= window_size
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
            alpha= alpha,
            zorder = zorders[i]
        )

    plt.legend(
        fontsize='large',
        labelspacing=0.2,
        handletextpad=0.5,
        borderaxespad=0.5,
        loc='lower right'
    ).set_zorder(10)


plt.savefig(image_path / f"lambda_{current_qoe}.svg")




