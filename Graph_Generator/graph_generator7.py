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
'''fixedUE 跟 movingUE 的 Hard slicing training curve
用以說明 movingUE 環境的波動。
'''

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Hyperparameters
algo_name = "Hard_Slicing"


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'utility'

fixed_csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/Hard_Slicing_csv/utility.csv")
moving_csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/Hard_Slicing_csv/utility.csv")
image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Combine") / f"HS_TrainingCurve"

fixed_utility = pd.read_csv(fixed_csv_path)['Value']
moving_utility = pd.read_csv(moving_csv_path)['Value']

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# plot figure


plt.figure(1)
plt.clf()
plt.title('HS_trainingCurve_2_env')
plt.xlabel('Episode')
plt.ylabel('utility')
plt.plot(ema(fixed_utility, weight= 0.9), label= 'fixed_UE', color= 'tab:green', alpha= 1)
plt.plot(ema(moving_utility, weight= 0.9), label= 'moving_UE', color= 'tab:red', alpha= 1)
plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
)
plt.savefig(image_path)









