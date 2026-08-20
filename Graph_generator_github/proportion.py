import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# used in generating the outcome figure
# np.convolve(data, kernel= np.ones(window_size) / window_size, mode= 'valid')，用 kernel 掃過整個 data (stride = 1)
# kernel : if window_size = 3, then kernel = [1/3, 1/3, 1/3]. 可以想成是每一個資料所佔的比例
# mode= 'valid'，不做 padding，只對完整的 window 做 moving average
def moving_average(data, window_size):
    data = np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode= 'valid')


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''Hyperparameters'''

color_1 = 'tab:red'
color_2 = 'tab:green'
color_3 = 'tab:blue'
zorders = [1, 0, 0]
seeds = [124]
steps = np.arange(9801)
Figure = "Test_Figures"
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#




csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[0]}"
image_path = Path(f"{Figure}/proportion")
# Automatically create the output directory if it does not exist
image_path.mkdir(parents=True, exist_ok=True)

algo_name1 = "GenDAC"
algo_name2 = "GANDDQN"
algo_name3 = "LSTM_A2C"
algo_name4 = "PPO"
algo_name5 = "SAC"
algo_names = [algo_name1, algo_name2, algo_name3, algo_name4, algo_name5]




for algo_name in algo_names:

    volte = moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "action_volte.csv")['Value'], window_size= 200) / 10**7
    embb = moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "action_embb_general.csv")['Value'], window_size= 200) / 10**7
    urllc = moving_average(pd.read_csv(csv_path / f"{algo_name}_csv" / "action_urllc.csv")['Value'], window_size= 200) / 10**7

    plt.figure(0)
    plt.clf()
    plt.title('Proportion of each slice')
    plt.xlabel('Decision Window')
    plt.ylabel('Proportion')

    # volte
    plt.plot(volte, label= 'VoLTE', color=color_1, linewidth=1.5)

    # embb
    plt.plot(embb, label= 'eMBB', color=color_2, linewidth=1.5)

    # urllc
    plt.plot(urllc, label= 'URLLC', color=color_3, linewidth=1.5)

    plt.legend(
        fontsize='medium',
        labelspacing=0.2,
        handletextpad=0.5,
        borderaxespad=0.5
        # loc='lower right'
    ).set_zorder(10)

    plt.savefig(image_path / f"{algo_name}.svg")