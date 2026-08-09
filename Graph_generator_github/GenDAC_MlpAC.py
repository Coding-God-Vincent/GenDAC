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
color_wr = 'tab:red'
color_wor = 'tab:green'
seeds = [124, 125, 126, 127, 128]
Figure = "Test_Figures"



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''1. Utility Learning Curve'''

algo1 = "GenDAC"
algo2 = "MlpAC"

colors = [color_wr, color_wor]
algo_names = [algo1, algo2]
labels = ["GenDAC (Diffusion Actor)", "MlpAC (MLP Actor)"]
steps = np.arange(9801)
means_across_algos = []
stds_across_algos = []
image_path = Path(f"{Figure}/GenDAC_MlpAC")

# calculate std & mean of each algo
for i, algo_name in enumerate(algo_names):
    values = []
    for j in range(len(seeds)):
        # set csv path
        csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
        values.append(moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "utility.csv")['Value'], window_size= 200))  # (5, 10000)
    values = np.array(values)
    means_across_algos.append(np.mean(values, axis= 0))  # (2, 10000)
    stds_across_algos.append(np.std(values, axis= 0))  # (2, 10000)

# generate figure
plt.figure(0)
plt.xlabel("Decision Window")
plt.ylabel("System Utility")
plt.title("Training Utility under Different Actor Design")

for i in range(len(algo_names)):
    upper = means_across_algos[i] + stds_across_algos[i]
    lower = means_across_algos[i] - stds_across_algos[i]
    plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], linewidth= 1.4, alpha= 1)
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
    fontsize= 'large',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
    loc= 'lower right'
).set_zorder(10)


plt.savefig(image_path / f"GenDAC_MlpAC_moving_utility.svg")



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''2. SE Learning Curve'''


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
    means_across_algos.append(np.mean(values, axis= 0))  # (2, 10000)
    stds_across_algos.append(np.std(values, axis= 0))  # (2, 10000)

# generate figure
plt.clf()
plt.figure(0)
plt.xlabel("Decision Window")
plt.ylabel("SE (bps/Hz)")
plt.title("SE under Different Actor Design")

for i in range(len(algo_names)):
    upper = means_across_algos[i] + stds_across_algos[i]
    lower = means_across_algos[i] - stds_across_algos[i]
    plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], linewidth= 1.4, alpha= 1)
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
    fontsize= 'large',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
    loc= 'lower right'
).set_zorder(10)

plt.savefig(image_path / f"GenDAC_MlpAC_moving_SE.svg")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''3. SSR'''

# qoe_names = ['qoe_embb_general', 'qoe_urllc', 'qoe_volte']
# markers = ['*', 'x']
# markereverys = [500, 600]

# for qoe_name in qoe_names:

#     if qoe_name == 'qoe_embb_general': title_name = 'SSR of eMBB service'
#     elif qoe_name == 'qoe_urllc' : title_name = 'SSR of URLLC service'
#     else: title_name = 'SSR of VoLTE service'

#     if fixed:
#         algo1 = "GenDAC_DDPM_1"
#         algo2 = "MlpAC"
#     else:
#         algo1 = "GenDAC_DDPM_3"
#         algo2 = "MlpAC"

#     colors = [color_wr, color_wor]
#     algo_names = [algo1, algo2]
#     labels = ["GenDAC (Diffusion Actor)", "MlpAC (MLP Actor)"]
#     steps = np.arange(10000)
#     means_across_algos = []
#     stds_across_algos = []
#     if fixed: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/Figures_scen2/others/GenDAC_MlpAC")
#     else: image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/Figures_scen2/others/GenDAC_MlpAC")

#     # calculate std & mean of each algo
#     for i, algo_name in enumerate(algo_names):
#         values = []
#         for j in range(len(seeds)):
#             # set csv path
#             if fixed: csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/CSVs") / f"seed_{seeds[j]}"
#             else: csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/CSVs") / f"seed_{seeds[j]}"
#             values.append(ema(pd.read_csv(csv_path / f"{algo_name}_csv" / f"{qoe_name}.csv")['Value'], weight= 0.9))  # (5, 10000)
#             # values.append(pd.read_csv(csv_path / f"{algo_name}_csv" / f"{qoe_name}.csv")['Value'])  # (5, 10000)
#         values = np.array(values)
#         means_across_algos.append(np.mean(values, axis= 0))  # (2, 10000)
#         stds_across_algos.append(np.std(values, axis= 0))  # (2, 10000)

#     # generate figure
#     plt.figure(0)
#     plt.clf()
#     plt.xlabel("Decision Window")
#     plt.ylabel("SSR")
#     plt.title(f"{title_name} under Different Actor Parameterizations")
#     # plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#     plt.ylim(0.0, 1.05)
#     plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))

#     for i in range(len(algo_names)):
#         # upper = means_across_algos[i] + stds_across_algos[i]
#         # lower = means_across_algos[i] - stds_across_algos[i]
#         plt.plot(steps, means_across_algos[i], color= colors[i], label= labels[i], linewidth= 0.7, alpha= 1, marker= markers[i], markevery= markereverys[i], markersize= 3.5)
#         # plt.plot(upper, linewidth= 1.0, color= colors[i], alpha= 0.5)
#         # plt.plot(lower, linewidth= 1.0, color= colors[i], alpha= 0.5)
#         # plt.fill_between(
#         #     steps,
#         #     upper,
#         #     lower,
#         #     color= colors[i],
#         #     alpha= 0.3
#         #     # zorder = zorders[i]
#         # )
        
#     plt.legend(
#         fontsize= 'small',  # 字體大小
#         labelspacing= 0.2,  # 垂直標籤之間的間距
#         handletextpad= 0.5,  # 圖示與文字之間的間距
#         borderaxespad= 0.5,  # 圖例框與邊框的間距
#         # ncol= 1  # 2 : 橫向、1 : 垂直
#         loc= 'lower right'
#     ).set_zorder(10)

#     if fixed: plt.savefig(image_path / f"GenDAC_MlpAC_fixedUE_{qoe_name}.svg")
#     else: plt.savefig(image_path / f"GenDAC_MlpAC_moving_{qoe_name}.svg")