import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Exponential Moving Average, bigger the weight (0~1) smoother the line
def ema(values, weight= 0.97):
    values = np.asarray(values, dtype=float)
    smoothed = np.zeros_like(values)  # 創建一個跟 values 一樣大的 np.zeros
    last = values[0]
    for i, v in enumerate(values):
        last = weight * last + (1 - weight) * v
        smoothed[i] = last
    return smoothed

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''1. 看資源分配的比例跟需求是否貼合還是超出很多
    1. observationBits / SE : 考量 SINR 的所需 Hz 數，Unit : Hz
    2. 資源分配量 : Unit : Hz
    * time : start from 0
'''

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Hyperparameters
algo_name = "SAC"
fixed = True

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
if fixed: 
    csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine") / f"{algo_name}_csv"
    image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/others/demand_and_supply_bits/total") / f"{algo_name}"
else: 
    csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine") / f"{algo_name}_csv"
    image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/others/demand_and_supply_bits/total") / f"{algo_name}"

se_embb_path = csv_path / "individual_se_embb_general.csv"
se_urllc_path = csv_path / "individual_se_urllc.csv"
se_volte_path = csv_path / "individual_se_volte.csv"
ob_embb_path = csv_path / "observationBits_embb_general.csv"
ob_urllc_path = csv_path / "observationBits_urllc.csv"
ob_volte_path = csv_path / "observationBits_volte.csv"
ac_embb_path = csv_path / "action_embb_general.csv"
ac_urllc_path = csv_path / "action_urllc.csv"
ac_volte_path = csv_path / "action_volte.csv"

# SE
# [0] -> time 0, [1] -> time 1, ..., [9999] -> time 9999, length 10000
se_embb = pd.read_csv(se_embb_path)['Value']
se_urllc = pd.read_csv(se_urllc_path)['Value']
se_volte = pd.read_csv(se_volte_path)['Value']

# observationBits of the 3 network slices (unit : bits)
# [0] -> x, [1] -> time 0, [2] -> time 1, ... [9999] -> time 9998, length 10000
ob_embb = pd.read_csv(ob_embb_path)['Value']
ob_urllc = pd.read_csv(ob_urllc_path)['Value']
ob_volte = pd.read_csv(ob_volte_path)['Value']

# action : resource qunatity of the 3 network slices (unit : Hz)
ac_embb = pd.read_csv(ac_embb_path)['Value']
ac_urllc = pd.read_csv(ac_urllc_path)['Value']
ac_volte = pd.read_csv(ac_volte_path)['Value']

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 因為狀態是使用前一個時刻的 bits，為了與真實 action 貼合，這邊要取出各時間步的真實 bits (unit : bits)

# line 1 (unit : bits), length 9999
actual_ob_embb = ob_embb[1 : 10000].reset_index(drop= True)  # time 0~9998 的真實 observation bits, [0] -> time 0, [1] -> time 1, ..., [9998] -> time 9998, length 9999
actual_ob_urllc = ob_urllc[1 : 10000].reset_index(drop= True)
actual_ob_volte = ob_volte[1 : 10000].reset_index(drop= True)

# line 2 (unit : bits), length 9999
# (unit : bits/s/hz)
cut_se_embb = se_embb[0 : 9999].reset_index(drop= True)  # time 0~9998 的 SE, length 9999
cut_se_urllc = se_urllc[0 : 9999].reset_index(drop= True)  # time 0~9998 的 SE, length 9999
cut_se_volte = se_volte[0 : 9999].reset_index(drop= True)  # time 0~9998 的 SE, length 9999
# (unit : bits/s)
cut_ac_embb = ac_embb[0 : 9999].reset_index(drop= True)  # time 0~9998 的分配資源量, length 9999
cut_ac_urllc = ac_urllc[0 : 9999].reset_index(drop= True)  # time 0~9998 的分配資源量, length 9999
cut_ac_volte = ac_volte[0 : 9999].reset_index(drop= True)  # time 0~9998 的分配資源量, length 9999

supplybit_embb = cut_se_embb * cut_ac_embb
supplybit_urllc = cut_se_urllc * cut_ac_urllc
supplybit_volte = cut_se_volte * cut_ac_volte



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# plot figure

'''1. observation bits (各網路切片於一個 window 中要傳送的 bits 數) vs supply bits (Action * individual_se -> 各網路切片於該 window 能傳送出多少 bits 數)'''

# embb_general
plt.figure(0)
plt.clf()
plt.title('demand bits vs supply bits - embb_general')
plt.xlabel('Time (s)')
plt.ylabel('bits')
# 因為一開始會差很多，這邊秀出穩定的部分
plt.plot(ema(actual_ob_embb, weight= 0.9), label= 'demand bits', color= 'tab:green', alpha= 0.5)
plt.plot(ema(supplybit_embb, weight= 0.9), label= 'supply bits', color= 'tab:red', alpha= 0.5)
plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
)
plt.savefig(image_path / "embb_general")

# urllc
plt.figure(1)
plt.clf()
plt.title('demand bits vs supply bits - urllc')
plt.xlabel('Time (s)')
plt.ylabel('bits')
plt.plot(ema(actual_ob_urllc, weight= 0.9), label= 'demand bits', color= 'tab:green', alpha= 0.5)
plt.plot(ema(supplybit_urllc, weight= 0.9), label= 'supply bits', color= 'tab:red', alpha= 0.5)
plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
)
plt.savefig(image_path / "urllc")

# volte
plt.figure(2)
plt.clf()
plt.title('demand bits vs supply bits - volte')
plt.xlabel('Time (s)')
plt.ylabel('bits')
plt.plot(ema(actual_ob_volte, weight= 0.9), label= 'demand bits', color= 'tab:green', alpha= 0.5)
plt.plot(ema(supplybit_volte, weight= 0.9), label= 'supply bits', color= 'tab:red', alpha= 0.5)
plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
)
plt.savefig(image_path / "volte")







