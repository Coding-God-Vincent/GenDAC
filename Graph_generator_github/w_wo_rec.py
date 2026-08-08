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
'''Hyperparameters'''
fixed = False
color_wr = 'tab:red'
color_wor = 'tab:green'
zorders = [1, 0]



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''1. Utility Learning Curve'''

if fixed:
    algo1 = "GenDAC_lam_05"
    algo2 = "GenDAC_lam_0"
else:
    algo1 = "GenDAC_different_speed"
    algo2 = "GenDAC_lam_0"

colors = [color_wr, color_wor]
algo_names = [algo1, algo2]
labels = [r"$\lambda \neq 0$", r"$\lambda = 0$"]
seeds = [124, 125, 126, 127, 128]
steps = np.arange(9801)
means_across_algos = []
stds_across_algos = []
if fixed: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/Figures_3_v2/max_action_3/others/w_wo_rec")
else: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/Figures_3_v2/max_action_3/others/w_wo_rec")

# calculate std & mean of each algo
for i, algo_name in enumerate(algo_names):
    values = []
    for j in range(len(seeds)):
        # set csv path
        if fixed: csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/CSVs") / f"seed_{seeds[j]}"
        else: csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/CSVs") / f"seed_{seeds[j]}"
        values.append(moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "utility.csv")['Value'], window_size= 200))  # (5, 10000)
    values = np.array(values)
    means_across_algos.append(np.mean(values, axis= 0))  # (2, 10000)
    stds_across_algos.append(np.std(values, axis= 0))  # (2, 10000)

# generate figure
plt.figure(0)
plt.clf()
plt.xlabel("Decision Window")
plt.ylabel("System Utility")
plt.title("Training Utility")

for i in range(len(algo_names)):
    upper = means_across_algos[i] + stds_across_algos[i]
    lower = means_across_algos[i] - stds_across_algos[i]
    plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], linewidth= 1.4, alpha= 1, zorder = zorders[i])
    # plt.plot(upper, linewidth= 1.0, color= colors[i], alpha= 0.5)
    # plt.plot(lower, linewidth= 1.0, color= colors[i], alpha= 0.5)
    plt.fill_between(
        steps,
        upper,
        lower,
        color= colors[i],
        alpha= 0.1,
        # zorder = zorders[i]
)
    
plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
    loc= 'lower right'
).set_zorder(10)

if fixed: plt.savefig(image_path / f"w_wo_rec_fixedUE_utility.svg")
else: plt.savefig(image_path / f"w_wo_rec_movingUE_utility.svg")



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''2. grad_norm_policy'''

if fixed:
    algo1 = "GenDAC_lam_001"
    algo2 = "GenDAC_3_wo_rec"
else:
    algo1 = "GenDAC_lam_001"
    algo2 = "GenDAC_3_wo_rec"

colors = [color_wr, color_wor]
algo_names = [algo1, algo2]
labels = [r"$\lambda \neq 0$", r"$\lambda = 0$"]
seeds = [124, 125, 126, 127, 128]
steps = np.arange(9802)
means_across_algos = []
stds_across_algos = []


# calculate std & mean of each algo
pad_len = 96  # warm up 時都不會有 grad_norm，所以補上 0
for i, algo_name in enumerate(algo_names):
    values = []
    for j in range(len(seeds)):
        # set csv path
        if fixed: csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/CSVs") / f"seed_{seeds[j]}"
        else: csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/CSVs") / f"seed_{seeds[j]}"
        values.append(np.concatenate([np.zeros(pad_len), moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "grad_grad_norm_policy.csv")['Value'], window_size= 200)]))  # (5, 10000)
        # values.append(np.concatenate([np.zeros(pad_len), pd.read_csv(csv_path / f"{algo_name}_csv" / "grad_grad_norm_policy.csv")['Value']]))  # (5, 10000) (without EMA)
    values = np.array(values)
    means_across_algos.append(np.mean(values, axis= 0))  # (2, 10000)
    stds_across_algos.append(np.std(values, axis= 0))  # (2, 10000)

# generate figure
plt.figure(0)
plt.clf()
plt.xlabel("Decision Window")
plt.ylabel("Grad Norm")
plt.title("Actor Gradient Norm (Critic-Induced)")
# plt.yscale("log")
plt.axhline(0.0, linestyle="--", alpha= 0.85, linewidth= 2, color= "k", label= "Gradient Threshold")  # 用虛線標示出有效梯度的界線


for i in range(len(algo_names)):
    upper = means_across_algos[i] + stds_across_algos[i]
    lower = means_across_algos[i] - stds_across_algos[i]
    lower = np.maximum(lower, 0)
    plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], linewidth= 1.4, alpha= 1)
    # plt.plot(upper, linewidth= 1.0, color= colors[i], alpha= 0.5)
    # plt.plot(lower, linewidth= 1.0, color= colors[i], alpha= 0.5)
    plt.fill_between(
        steps,
        upper,
        lower,
        color= colors[i],
        alpha= 0.1
        # zorder = zorders[i]
)
    
plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
    loc= 'upper right'
).set_zorder(10)

if fixed: plt.savefig(image_path / f"w_wo_rec_fixedUE_policy_grad_norm.svg")
else: plt.savefig(image_path / f"w_wo_rec_movingUE_policy_grad_norm.svg")