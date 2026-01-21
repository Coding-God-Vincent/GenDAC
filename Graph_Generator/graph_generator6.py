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

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''Hyperparameter'''
alpha = 0.05  # 透明度
weight = 0.97  # 平滑程度 (除 qoe 其他是 0.9)
color_d2ac_obj = 'tab:red'
color_d2ac_rew = 'tab:green'



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''Uitility'''
# steps = np.arange(10000)

# # 讀 csv 檔
# d2ac_obj_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_DDPM_1_csv/utility.csv")
# d2ac_rew_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_DDPM_1_r_csv/utility.csv")


# # 算出上下界
# smooth_d2ac_obj = ema(d2ac_obj_utility['Value'], weight= 0.9)
# lower_d2ac_obj = np.minimum(d2ac_obj_utility['Value'], smooth_d2ac_obj)
# upper_d2ac_obj = np.maximum(d2ac_obj_utility['Value'], smooth_d2ac_obj)

# smooth_d2ac_rew = ema(d2ac_rew_utility['Value'], weight= 0.9)
# lower_d2ac_rew = np.minimum(d2ac_rew_utility['Value'], smooth_d2ac_rew)
# upper_d2ac_rew = np.maximum(d2ac_rew_utility['Value'], smooth_d2ac_rew)


# # 畫圖
# plt.figure(0)
# plt.clf()
# plt.title('Utility')
# plt.xlabel('Episode')
# plt.ylabel('utility')
# # D2AC_P1_obj
# plt.plot(smooth_d2ac_obj, label= 'D2AC_obj', color= color_d2ac_obj, zorder= 7)
# plt.fill_between(x= steps, y1= lower_d2ac_obj, y2= lower_d2ac_obj, color= color_d2ac_obj, alpha= alpha)
# # D2AC_P1_rew
# plt.plot(smooth_d2ac_rew, label= 'D2AC_rew', color= color_d2ac_rew, zorder= 4)
# plt.fill_between(x= steps, y1= lower_d2ac_rew, y2= upper_d2ac_rew, color= color_d2ac_rew, alpha= alpha)

# plt.legend(
#     fontsize= 'small',  # 字體大小
#     labelspacing= 0.2,  # 垂直標籤之間的間距
#     handletextpad= 0.5,  # 圖示與文字之間的間距
#     borderaxespad= 0.5,  # 圖例框與邊框的間距
#     # ncol= 1  # 2 : 橫向、1 : 垂直
#     loc= 'lower left'
# ).set_zorder(10)
# plt.savefig('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/others/D2AC_obj_and_rew/Utility')


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''SE'''
# steps = np.arange(10000)

# # 讀 csv 檔
# d2ac_obj_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_DDPM_1_csv/se.csv")
# d2ac_rew_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_DDPM_1_r_csv/se.csv")


# # 算出上下界
# smooth_d2ac_obj = ema(d2ac_obj_se['Value'], weight= 0.9)
# lower_d2ac_obj = np.minimum(d2ac_obj_se['Value'], smooth_d2ac_obj)
# upper_d2ac_obj = np.maximum(d2ac_obj_se['Value'], smooth_d2ac_obj)

# smooth_d2ac_rew = ema(d2ac_rew_se['Value'], weight= 0.9)
# lower_d2ac_rew = np.minimum(d2ac_rew_se['Value'], smooth_d2ac_rew)
# upper_d2ac_rew = np.maximum(d2ac_rew_se['Value'], smooth_d2ac_rew)


# # 畫圖
# plt.figure(0)
# plt.clf()
# plt.title('SE')
# plt.xlabel('Episode')
# plt.ylabel('SE')
# # D2AC_P1_obj
# plt.plot(smooth_d2ac_obj, label= 'D2AC_obj', color= color_d2ac_obj, zorder= 7)
# plt.fill_between(x= steps, y1= lower_d2ac_obj, y2= lower_d2ac_obj, color= color_d2ac_obj, alpha= alpha)
# # D2AC_P1_rew
# plt.plot(smooth_d2ac_rew, label= 'D2AC_rew', color= color_d2ac_rew, zorder= 4)
# plt.fill_between(x= steps, y1= lower_d2ac_rew, y2= upper_d2ac_rew, color= color_d2ac_rew, alpha= alpha)

# plt.legend(
#     fontsize= 'small',  # 字體大小
#     labelspacing= 0.2,  # 垂直標籤之間的間距
#     handletextpad= 0.5,  # 圖示與文字之間的間距
#     borderaxespad= 0.5,  # 圖例框與邊框的間距
#     # ncol= 1  # 2 : 橫向、1 : 垂直
#     loc= 'lower left'
# ).set_zorder(10)
# plt.savefig('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/others/D2AC_obj_and_rew/SE')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''QoE'''
qoes = ['volte', 'urllc', 'embb_general']
steps = np.arange(10000)

for qoe in qoes:
    # 讀 csv 檔
    obj_qoe_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_DDPM_1_csv") / f"qoe_{qoe}.csv" 
    rew_qoe_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_DDPM_1_r_csv") / f"qoe_{qoe}.csv" 
    image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/others/D2AC_obj_and_rew") / f"{qoe}"

    d2ac_obj_qoe = pd.read_csv(obj_qoe_path)['Value']
    d2ac_rew_qoe = pd.read_csv(rew_qoe_path)['Value']

    # 為避免超過 1 (前一時刻沒傳完的若這一刻傳完會記到當前時刻)
    d2ac_obj_qoe = np.clip(d2ac_obj_qoe, 0.0, 1.0)
    d2ac_rew_qoe = np.clip(d2ac_rew_qoe, 0.0, 1.0)
    

    # 算出上下界
    smooth_d2ac_obj = ema(d2ac_obj_qoe, weight= 0.9)
    lower_d2ac_obj = np.minimum(d2ac_obj_qoe, smooth_d2ac_obj)
    upper_d2ac_obj = np.maximum(d2ac_obj_qoe, smooth_d2ac_obj)

    smooth_d2ac_rew = ema(d2ac_rew_qoe, weight= 0.9)
    lower_d2ac_rew = np.minimum(d2ac_rew_qoe, smooth_d2ac_rew)
    upper_d2ac_rew = np.maximum(d2ac_rew_qoe, smooth_d2ac_rew)


    # 畫圖
    plt.figure(0)
    plt.clf()
    name = qoe
    if qoe == 'embb_general': name = 'Video'
    plt.title(name)
    plt.xlabel('Episode')
    plt.ylabel('QoE')
    # D2AC_P1_obj
    plt.plot(smooth_d2ac_obj, label= 'D2AC_obj', color= color_d2ac_obj, zorder= 7)
    plt.fill_between(x= steps, y1= lower_d2ac_obj, y2= lower_d2ac_obj, color= color_d2ac_obj, alpha= alpha)
    # D2AC_P1_rew
    plt.plot(smooth_d2ac_rew, label= 'D2AC_rew', color= color_d2ac_rew, zorder= 4)
    plt.fill_between(x= steps, y1= lower_d2ac_rew, y2= upper_d2ac_rew, color= color_d2ac_rew, alpha= alpha)

    plt.legend(
        fontsize= 'small',  # 字體大小
        labelspacing= 0.2,  # 垂直標籤之間的間距
        handletextpad= 0.5,  # 圖示與文字之間的間距
        borderaxespad= 0.5,  # 圖例框與邊框的間距
        # ncol= 1  # 2 : 橫向、1 : 垂直
        loc= 'lower left'
    ).set_zorder(10)

    plt.savefig(image_path)