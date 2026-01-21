from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Hyperparameters
algo_name = "D2AC_DDPM_1"
fixed = True

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
if fixed: 
    csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine") / f"{algo_name}_csv"
    image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/others/demand_and_supply_bits") / f"{algo_name}"
else: 
    csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine") / f"{algo_name}_csv"
    image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/others/demand_and_supply_bits") / f"{algo_name}"

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

ob_embb_path = csv_path / "observationBits_embb_general.csv"
ob_urllc_path = csv_path / "observationBits_urllc.csv"
ob_volte_path = csv_path / "observationBits_volte.csv"

# observationBits of the 3 network slices (unit : bits)
# [0] -> x, [1] -> time 0, [2] -> time 1, ... [9999] -> time 9998, length 10000
ob_embb = pd.read_csv(ob_embb_path)['Value']
ob_urllc = pd.read_csv(ob_urllc_path)['Value']
ob_volte = pd.read_csv(ob_volte_path)['Value']

# (unit : bits), length 9999
actual_ob_embb = ob_embb[1 : 10000].reset_index(drop= True)  # time 0~9998 的真實 observation bits, [0] -> time 0, [1] -> time 1, ..., [9998] -> time 9998, length 9999
actual_ob_urllc = ob_urllc[1 : 10000].reset_index(drop= True)
actual_ob_volte = ob_volte[1 : 10000].reset_index(drop= True)

'''2. observation bits'''
# embb_general
plt.figure(6)
plt.clf()
plt.title('Observation bits of embb-general')
plt.xlabel('Time (s)')
plt.ylabel('Bits')
plt.plot(actual_ob_embb, label= 'Observation bits', color= 'tab:green')
plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/others/Observation_bits/embb_general")

# urllc
plt.figure(7)
plt.clf()
plt.title('Observation bits of urllc')
plt.xlabel('Time (s)')
plt.ylabel('Bits')
plt.plot(actual_ob_urllc, label= 'Observation bits', color= 'tab:green')
plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/others/Observation_bits/urllc")

# volte
plt.figure(8)
plt.clf()
plt.title('Observation bits of volte')
plt.xlabel('Time (s)')
plt.ylabel('Bits')
plt.plot(actual_ob_volte, label= 'Observation bits', color= 'tab:green')
plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/others/Observation_bits/volte")