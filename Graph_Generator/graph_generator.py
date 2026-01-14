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
color_d2ac = 'tab:red'
color_ganddqn = 'tab:green'
color_hard_slicing = 'tab:blue'
color_lstm_a2c = 'tab:orange'
color_ppo = 'tab:pink'
color_sac = 'tab:cyan'


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''Uitility'''
# steps = np.arange(10000)

# # 讀 csv 檔
# d2ac_P1_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/D2AC_DDPM_1_csv/utility.csv")
# ganddqn_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/GANDDQN_csv/utility.csv")
# hard_slicing_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/Hard_Slicing_csv/utility.csv")
# lstm_a2c_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/LSTM_A2C_csv/utility.csv")
# ppo_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/PPO_csv/utility.csv")
# sac_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/SAC_csv/utility.csv")

# # 算出上下界
# smooth_d2ac_P1 = ema(d2ac_P1_utility['Value'], weight= 0.9)
# lower_d2ac_P1 = np.minimum(d2ac_P1_utility['Value'], smooth_d2ac_P1)
# upper_d2ac_P1 = np.maximum(d2ac_P1_utility['Value'], smooth_d2ac_P1)

# smooth_ganddqn = ema(ganddqn_utility['Value'], weight= 0.9)
# lower_ganddqn = np.minimum(ganddqn_utility['Value'], smooth_ganddqn)
# upper_ganddqn = np.maximum(ganddqn_utility['Value'], smooth_ganddqn)

# smooth_hard_slicing = ema(hard_slicing_utility['Value'], weight= 0.9)
# lower_hard_slicing = np.minimum(hard_slicing_utility['Value'], smooth_hard_slicing)
# upper_hard_slicing = np.maximum(hard_slicing_utility['Value'], smooth_hard_slicing)

# smooth_lstm_a2c = ema(lstm_a2c_utility['Value'], weight= 0.9)
# lower_lstm_a2c = np.minimum(lstm_a2c_utility['Value'], smooth_lstm_a2c)
# upper_lstm_a2c = np.maximum(lstm_a2c_utility['Value'], smooth_lstm_a2c)

# smooth_ppo = ema(ppo_utility['Value'], weight= 0.9)
# lower_ppo = np.minimum(ppo_utility['Value'], smooth_ppo)
# upper_ppo = np.maximum(ppo_utility['Value'], smooth_ppo)

# smooth_sac = ema(sac_utility['Value'], weight= 0.9)
# lower_sac = np.minimum(sac_utility['Value'], smooth_sac)
# upper_sac = np.maximum(sac_utility['Value'], smooth_sac)

# # 畫圖
# plt.figure(0)
# plt.clf()
# plt.title('Utility')
# plt.xlabel('Episode')
# plt.ylabel('utility')
# # D2AC_P1
# plt.plot(smooth_d2ac_P1, label= 'D2AC_P1', color= color_d2ac, zorder= 7)
# plt.fill_between(x= steps, y1= lower_d2ac_P1, y2= upper_d2ac_P1, color= color_d2ac, alpha= alpha)
# # GANDDQN
# plt.plot(smooth_ganddqn, label= 'GANDDQN', color= color_ganddqn, zorder= 4)
# plt.fill_between(x= steps, y1= lower_ganddqn, y2= upper_ganddqn, color= color_ganddqn, alpha= alpha)
# # Hard_Slicing
# plt.plot(smooth_hard_slicing, label= 'Hard Slicing', color= color_hard_slicing, zorder= 5)
# plt.fill_between(x= steps, y1= lower_hard_slicing, y2= upper_hard_slicing, color= color_hard_slicing, alpha= alpha)
# # LSTM-A2C
# plt.plot(smooth_lstm_a2c, label= 'LSTM-A2C', color= color_lstm_a2c, zorder= 6)
# plt.fill_between(x= steps, y1= lower_lstm_a2c, y2= upper_lstm_a2c, color= color_lstm_a2c, alpha= alpha)
# # PPO
# plt.plot(smooth_ppo, label= 'PPO', color= color_ppo)
# plt.fill_between(x= steps, y1= lower_ppo, y2= upper_ppo, color= color_ppo, alpha= alpha)
# # SAC
# plt.plot(smooth_sac, label= 'SAC', color= color_sac)
# plt.fill_between(x= steps, y1= lower_sac, y2= upper_sac, color= color_sac, alpha= alpha)
# plt.legend(
#     fontsize= 'small',  # 字體大小
#     labelspacing= 0.2,  # 垂直標籤之間的間距
#     handletextpad= 0.5,  # 圖示與文字之間的間距
#     borderaxespad= 0.5,  # 圖例框與邊框的間距
#     # ncol= 1  # 2 : 橫向、1 : 垂直
# )
# plt.savefig('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/6_algos/Utility_fixedUE')


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''SE'''
# steps = np.arange(10000)

# # 讀 csv 檔
# d2ac_P1_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/D2AC_DDPM_1_csv/se.csv")
# ganddqn_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/GANDDQN_csv/se.csv")
# hard_slicing_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/Hard_Slicing_csv/se.csv")
# lstm_a2c_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/LSTM_A2C_csv/se.csv")
# ppo_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/PPO_csv/se.csv")
# sac_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/SAC_csv/se.csv")

# # 算出上下界
# smooth_d2ac_P1 = ema(d2ac_P1_se['Value'], weight= 0.9)
# lower_d2ac_P1 = np.minimum(d2ac_P1_se['Value'], smooth_d2ac_P1)
# upper_d2ac_P1 = np.maximum(d2ac_P1_se['Value'], smooth_d2ac_P1)

# smooth_ganddqn = ema(ganddqn_se['Value'], weight= 0.9)
# lower_ganddqn = np.minimum(ganddqn_se['Value'], smooth_ganddqn)
# upper_ganddqn = np.maximum(ganddqn_se['Value'], smooth_ganddqn)

# smooth_hard_slicing = ema(hard_slicing_se['Value'], weight= 0.9)
# lower_hard_slicing = np.minimum(hard_slicing_se['Value'], smooth_hard_slicing)
# upper_hard_slicing = np.maximum(hard_slicing_se['Value'], smooth_hard_slicing)

# smooth_lstm_a2c = ema(lstm_a2c_se['Value'], weight= 0.9)
# lower_lstm_a2c = np.minimum(lstm_a2c_se['Value'], smooth_lstm_a2c)
# upper_lstm_a2c = np.maximum(lstm_a2c_se['Value'], smooth_lstm_a2c)

# smooth_ppo = ema(ppo_se['Value'], weight= 0.9)
# lower_ppo = np.minimum(ppo_se['Value'], smooth_ppo)
# upper_ppo = np.maximum(ppo_se['Value'], smooth_ppo)

# smooth_sac = ema(sac_se['Value'], weight= 0.9)
# lower_sac = np.minimum(sac_se['Value'], smooth_sac)
# upper_sac = np.maximum(sac_se['Value'], smooth_sac)

# # 畫圖
# plt.figure(0)
# plt.clf()
# plt.title('SE')
# plt.xlabel('Episode')
# plt.ylabel('SE')
# # D2AC_P1
# plt.plot(smooth_d2ac_P1, label= 'D2AC_P1', color= color_d2ac, zorder= 7)
# plt.fill_between(x= steps, y1= lower_d2ac_P1, y2= upper_d2ac_P1, color= color_d2ac, alpha= alpha)
# # GANDDQN
# plt.plot(smooth_ganddqn, label= 'GANDDQN', color= color_ganddqn, zorder= 4)
# plt.fill_between(x= steps, y1= lower_ganddqn, y2= upper_ganddqn, color= color_ganddqn, alpha= alpha)
# # Hard_Slicing
# plt.plot(smooth_hard_slicing, label= 'Hard Slicing', color= color_hard_slicing, zorder= 5)
# plt.fill_between(x= steps, y1= lower_hard_slicing, y2= upper_hard_slicing, color= color_hard_slicing, alpha= alpha)
# # LSTM-A2C
# plt.plot(smooth_lstm_a2c, label= 'LSTM-A2C', color= color_lstm_a2c, zorder= 6)
# plt.fill_between(x= steps, y1= lower_lstm_a2c, y2= upper_lstm_a2c, color= color_lstm_a2c, alpha= alpha)
# # PPO
# plt.plot(smooth_ppo, label= 'PPO', color= color_ppo)
# plt.fill_between(x= steps, y1= lower_ppo, y2= upper_ppo, color= color_ppo, alpha= alpha)
# # SAC
# plt.plot(smooth_sac, label= 'SAC', color= color_sac)
# plt.fill_between(x= steps, y1= lower_sac, y2= upper_sac, color= color_sac, alpha= alpha)

# plt.legend(
#     fontsize= 'small',  # 字體大小
#     labelspacing= 0.2,  # 垂直標籤之間的間距
#     handletextpad= 0.5,  # 圖示與文字之間的間距
#     borderaxespad= 0.5,  # 圖例框與邊框的間距
#     # ncol= 1  # 2 : 橫向、1 : 垂直
# )
# plt.savefig('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/6_algos/SE_fixedUE')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''QoE'''

# # Hyperparameters
# current_qoe = 'qoe_volte'  # qoe_embb_general, qoe_urllc, qoe_volte
# fixed= False
# algo1 = 'D2AC_DDPM_1'
# algo2 = 'GANDDQN'
# algo3 = 'Hard_Slicing'
# algo4 = 'LSTM_A2C'
# algo5 = 'PPO'
# algo6 = 'SAC'


# #-----------------------------------------------------------------------------------------------------------------------------------------------#
# if current_qoe == 'qoe_embb_general': title = 'video service'
# elif current_qoe == 'qoe_urllc' : title = 'URLLC service'
# else: title = "VoLTE service"

# if fixed : 
#     combine_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
#     image_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/6_algos') / f"{current_qoe}.png"
# else : 
#     combine_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine')
#     image_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/6_algos') / f"{current_qoe}.png"

# steps = np.arange(10000)

# d2ac_P1_path = combine_path / f"{algo1}_csv" / f"{current_qoe}.csv"
# ganddqn_path = combine_path / f"{algo2}_csv" / f"{current_qoe}.csv"
# hard_slicing_path = combine_path / f"{algo3}_csv" / f"{current_qoe}.csv"
# lstm_a2c_path = combine_path / f"{algo4}_csv" / f"{current_qoe}.csv"
# ppo_path = combine_path / f"{algo5}_csv" / f"{current_qoe}.csv"
# sac_path = combine_path / f"{algo6}_csv" / f"{current_qoe}.csv"

# # 讀取 csv 檔
# d2ac_P1_qoe = pd.read_csv(d2ac_P1_path)
# ganddqn_qoe = pd.read_csv(ganddqn_path)
# hard_slicing_qoe = pd.read_csv(hard_slicing_path)
# lstm_a2c_qoe = pd.read_csv(lstm_a2c_path)
# ppo_qoe = pd.read_csv(ppo_path)
# sac_qoe = pd.read_csv(sac_path)

# # 算出上下界
# smooth_d2ac_P1 = ema(d2ac_P1_qoe['Value'], weight= weight)
# lower_d2ac_P1 = np.minimum(d2ac_P1_qoe['Value'], smooth_d2ac_P1)
# upper_d2ac_P1 = np.maximum(d2ac_P1_qoe['Value'], smooth_d2ac_P1)

# smooth_ganddqn = ema(ganddqn_qoe['Value'], weight= weight)
# lower_ganddqn = np.minimum(ganddqn_qoe['Value'], smooth_ganddqn)
# upper_ganddqn = np.maximum(ganddqn_qoe['Value'], smooth_ganddqn)

# smooth_hard_slicing = ema(hard_slicing_qoe['Value'], weight= weight)
# lower_hard_slicing = np.minimum(hard_slicing_qoe['Value'], smooth_hard_slicing)
# upper_hard_slicing = np.maximum(hard_slicing_qoe['Value'], smooth_hard_slicing)

# smooth_lstm_a2c = ema(lstm_a2c_qoe['Value'], weight= weight)
# lower_lstm_a2c = np.minimum(lstm_a2c_qoe['Value'], smooth_lstm_a2c)
# upper_lstm_a2c = np.maximum(lstm_a2c_qoe['Value'], smooth_lstm_a2c)

# smooth_ppo = ema(ppo_qoe['Value'], weight= weight)
# lower_ppo = np.minimum(ppo_qoe['Value'], smooth_ppo)
# upper_ppo = np.maximum(ppo_qoe['Value'], smooth_ppo)

# smooth_sac = ema(sac_qoe['Value'], weight= weight)
# lower_sac = np.minimum(sac_qoe['Value'], smooth_sac)
# upper_sac = np.maximum(sac_qoe['Value'], smooth_sac)

# # 畫圖
# plt.figure(0)
# plt.clf()
# plt.title(title)
# plt.xlabel('Episode')
# plt.ylabel('SSR')
# # D2AC_P1
# plt.plot(smooth_d2ac_P1, label= 'D2AC_P1', color= color_d2ac, zorder= 7)
# plt.fill_between(x= steps, y1= lower_d2ac_P1, y2= upper_d2ac_P1, color= color_d2ac, alpha= alpha)
# # GANDDQN
# plt.plot(smooth_ganddqn, label= 'GANDDQN', color= color_ganddqn, zorder= 4)
# plt.fill_between(x= steps, y1= lower_ganddqn, y2= upper_ganddqn, color= color_ganddqn, alpha= alpha)
# # Hard_Slicing
# plt.plot(smooth_hard_slicing, label= 'Hard Slicing', color= color_hard_slicing, zorder= 5)
# plt.fill_between(x= steps, y1= lower_hard_slicing, y2= upper_hard_slicing, color= color_hard_slicing, alpha= alpha)
# # LSTM-A2C
# plt.plot(smooth_lstm_a2c, label= 'LSTM-A2C', color= color_lstm_a2c, zorder= 6)
# plt.fill_between(x= steps, y1= lower_lstm_a2c, y2= upper_lstm_a2c, color= color_lstm_a2c, alpha= alpha)
# # PPO
# plt.plot(smooth_ppo, label= 'PPO', color= color_ppo)
# plt.fill_between(x= steps, y1= lower_ppo, y2= upper_ppo, color= color_ppo, alpha= alpha)
# # SAC
# plt.plot(smooth_sac, label= 'SAC', color= color_sac)
# plt.fill_between(x= steps, y1= lower_sac, y2= upper_sac, color= color_sac, alpha= alpha)

# plt.legend(
#     fontsize= 'small',  # 字體大小
#     labelspacing= 0.2,  # 垂直標籤之間的間距
#     handletextpad= 0.5,  # 圖示與文字之間的間距
#     borderaxespad= 0.5,  # 圖例框與邊框的間距
#     # ncol= 1  # 2 : 橫向、1 : 垂直
# )
# plt.savefig(image_path)




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''ObservationBits'''

# # Hyperparameters
# Fixed = True
# ns1 = 'embb_general'  # embb_general, urllc, volte
# ns2 = 'urllc'
# ns3 = 'volte'


# #-----------------------------------------------------------------------------------------------------------------------------------------------#
# steps = np.arange(10000)

# if Fixed : 
#     combine_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
#     image_path = combine_path / f"fixedUE_observation_bits"
#     title = "fixedUE_observation_bits"
# else: 
#     combine_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine')
#     image_path = combine_path / f"movingUE_observation_bits"
#     title = "movingUE_observation_bits"

# embb_path = Path(combine_path / f"D2AC_DDPM_1_csv" / f"observationBits_{ns1}.csv")
# urllc_path = Path(combine_path / f"D2AC_DDPM_1_csv" / f"observationBits_{ns1}.csv")
# volte_path = Path(combine_path / f"D2AC_DDPM_1_csv" / f"observationBits_{ns1}.csv")

# embb = pd.read_csv(embb_path)
# urllc = pd.read_csv(urllc_path)
# volte = pd.read_csv(volte_path)

# smooth_embb = ema(embb['Value'], weight= 0.9)
# lower_embb = np.minimum(embb['Value'], smooth_embb)
# upper_embb = np.maximum(embb['Value'], smooth_embb)

# smooth_urllc = ema(urllc['Value'], weight= 0.9)
# lower_urllc = np.minimum(urllc['Value'], smooth_urllc)
# upper_urllc = np.maximum(urllc['Value'], smooth_urllc)

# smooth_volte = ema(volte['Value'], weight= 0.9)
# lower_volte = np.minimum(volte['Value'], smooth_volte)
# upper_volte = np.maximum(volte['Value'], smooth_volte)


# plt.figure(2)
# plt.clf()
# plt.title(title)
# plt.xlabel('Episode')
# plt.ylabel('Bits')
# plt.plot(smooth_embb, label= 'video', color= 'tab:orange')
# plt.fill_between(x= steps, y1= lower_embb, y2= upper_embb, color= 'orange', alpha= 0.15)
# plt.plot(smooth_urllc, label= 'urllc', color= 'tab:green')
# plt.fill_between(x= steps, y1= lower_urllc, y2= upper_urllc, color= 'green', alpha= 0.15)
# plt.plot(smooth_volte, label= 'volte', color= 'tab:blue')
# plt.fill_between(x= steps, y1= lower_volte, y2= upper_volte, color= 'blue', alpha= 0.15)
# plt.legend()
# plt.savefig(image_path)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''SE : D2AC_P1 vs D2AC_P6'''
# # hyperparameters
# Fixed = False

# #-----------------------------------------------------------------------------------------------------------------------------------------------#
# steps = np.arange(10000)
# demo_steps = 1000

# if Fixed:
#     combine_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
#     image_path = combine_path / f"6_algos" / f"SE_DP6_DP1"
# else: 
#     combine_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine')
#     image_path = combine_path / f"6_algos" / f"SE_DP6_DP1"

# # 讀 csv 檔
# d2ac_P1_se = pd.read_csv(combine_path / f"D2AC_DDPM_1_csv" / f"se.csv")
# d2ac_P6_se = pd.read_csv(combine_path / f"D2AC_DDPM_6_csv" / f"se.csv")

# # 算出上下界
# smooth_d2ac_P1 = ema(d2ac_P1_se['Value'], weight= 0.9)
# lower_d2ac_P1 = np.minimum(d2ac_P1_se['Value'], smooth_d2ac_P1)
# upper_d2ac_P1 = np.maximum(d2ac_P1_se['Value'], smooth_d2ac_P1)

# smooth_d2ac_P6 = ema(d2ac_P6_se['Value'], weight= 0.9)
# lower_d2ac_P6 = np.minimum(d2ac_P6_se['Value'], smooth_d2ac_P6)
# upper_d2ac_P6 = np.maximum(d2ac_P6_se['Value'], smooth_d2ac_P6)

# # 畫圖
# plt.figure(0)
# plt.clf()
# plt.title('SE')
# plt.xlabel('Episode')
# plt.ylabel('SE')
# # D2AC_P1
# plt.plot(smooth_d2ac_P1[0:demo_steps], label= 'D2AC_P1', color= color_d2ac, zorder= 7)
# plt.fill_between(x= demo_steps, y1= lower_d2ac_P1[0:demo_steps], y2= upper_d2ac_P1[0:demo_steps], color= color_d2ac, alpha= alpha)
# # D2AC_P6
# plt.plot(smooth_d2ac_P6[0:demo_steps], label= 'D2AC_P6', color= color_ganddqn, zorder= 4)
# plt.fill_between(x= demo_steps, y1= lower_d2ac_P6[0:demo_steps], y2= upper_d2ac_P6[0:demo_steps], color= color_ganddqn, alpha= alpha)


# plt.legend(
#     fontsize= 'small',  # 字體大小
#     labelspacing= 0.2,  # 垂直標籤之間的間距
#     handletextpad= 0.5,  # 圖示與文字之間的間距
#     borderaxespad= 0.5,  # 圖例框與邊框的間距
#     # ncol= 1  # 2 : 橫向、1 : 垂直
# )

# plt.savefig(image_path)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''Utility : D2AC_P1 vs D2AC_P6'''

# hyperparameters
Fixed = False

#-----------------------------------------------------------------------------------------------------------------------------------------------#
steps = np.arange(10000)
demo_steps = 1000

if Fixed:
    combine_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
    image_path = combine_path / f"6_algos" / f"utility_DP6_DP1"
else: 
    combine_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine')
    image_path = combine_path / f"6_algos" / f"utility_DP6_DP1"

# 讀 csv 檔
d2ac_P1_utility = pd.read_csv(combine_path / f"D2AC_DDPM_1_csv" / f"utility.csv")
d2ac_P6_utility = pd.read_csv(combine_path / f"D2AC_DDPM_6_csv" / f"utility.csv")

# 算出上下界
smooth_d2ac_P1 = ema(d2ac_P1_utility['Value'], weight= 0.9)
lower_d2ac_P1 = np.minimum(d2ac_P1_utility['Value'], smooth_d2ac_P1)
upper_d2ac_P1 = np.maximum(d2ac_P1_utility['Value'], smooth_d2ac_P1)

smooth_d2ac_P6 = ema(d2ac_P6_utility['Value'], weight= 0.9)
lower_d2ac_P6 = np.minimum(d2ac_P6_utility['Value'], smooth_d2ac_P6)
upper_d2ac_P6 = np.maximum(d2ac_P6_utility['Value'], smooth_d2ac_P6)

# 畫圖
plt.figure(0)
plt.clf()
plt.title('Utility')
plt.xlabel('Episode')
plt.ylabel('Utility')
# D2AC_P1
plt.plot(smooth_d2ac_P1[0:demo_steps], label= 'D2AC_P1', color= color_d2ac, zorder= 7)
plt.fill_between(x= demo_steps, y1= lower_d2ac_P1[0:demo_steps], y2= upper_d2ac_P1[0:demo_steps], color= color_d2ac, alpha= alpha)
# D2AC_P6
plt.plot(smooth_d2ac_P6[0:demo_steps], label= 'D2AC_P6', color= color_ganddqn, zorder= 4)
plt.fill_between(x= demo_steps, y1= lower_d2ac_P6[0:demo_steps], y2= upper_d2ac_P6[0:demo_steps], color= color_ganddqn, alpha= alpha)


plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
)

plt.savefig(image_path)