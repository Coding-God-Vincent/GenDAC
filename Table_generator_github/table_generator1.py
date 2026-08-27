import numpy as np
import pandas as pd
from pathlib import Path

'''流程 (以 Utility 為例)

1. 各指標最後 1000 個值先做 EMA 平滑處理 -> Utility_1000
2. 計算單一 seeds 中 Utility_1000 的 mean & std -> 5 個 seeds 的 means & stds
3. 平均那 5 個 seeds 的 mean & std -> final mean & std

'''

# ===== EMA =====
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


# ===== helper function =====
def compute_mean_std(data_1000):
    """
    data_1000: shape (num_seeds, 1000)
    return: (mean, std)
    """
    # Step 1: 每個 seed 在最後 1000 steps 的 mean & std
    per_seed_mean = np.mean(data_1000, axis= 1)  # shape: (num_seeds,)
    per_seed_std = np.mean(data_1000, axis= 1)  # shape (num_seeds, )
    # per_seed_mean = np.mean(data_1000, axis=0)   # shape: (10000)

    # Step 2: across seeds 算 mean 和 std
    mean = np.mean(per_seed_mean)
    std = np.std(per_seed_std)

    return mean, std


# ===== 單一方法的統計 =====
def compute_method_stats(Utility, SE, eMBB, URLLC, VoLTE):
    stats = {}

    stats["Utility"] = compute_mean_std(Utility)
    stats["SE"] = compute_mean_std(SE)
    stats["eMBB SSR"] = compute_mean_std(eMBB)
    stats["URLLC SSR"] = compute_mean_std(URLLC)
    stats["VoLTE SSR"] = compute_mean_std(VoLTE)

    return stats


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''hyperparameters'''
fixed = False
seeds = [124, 125, 126, 127, 128]
# seeds = [124]
length = -1000



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# algo_names = ["GenDAC", "GANDDQN", "LSTM_A2C", "Hard_Slicing", "PPO", "SAC"]
algo_names = ["GenDAC_bs_1", "GenDAC", "GenDAC_bs_5"]
metrics = ["utility", "se", "qoe_embb_general", "qoe_urllc", "qoe_volte"]

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 存所有 Algo 的所有結果
results = {}

for i, algo_name in enumerate(algo_names):
    
    csv_root_path = Path("Outcome_github/CSVs")
    
    # 暫存五個 seeds 的各指標內容
    Utilities = []
    SEs = []
    Qoe_embbs = []
    Qoe_urllcs = []
    Qoe_volte = []

    # 將 5 個 seeds 經過 moving average 後的後面 1000 個值存起來 (5, 1000)
    for seed in seeds:
        # Utility
        Utilities.append(moving_average(pd.read_csv(csv_root_path / f"seed_{seed}" / f"{algo_name}_csv" / "utility.csv")['Value'], window_size= 200)[length:])  # np.array with (5, 1000)
        SEs.append(moving_average(pd.read_csv(csv_root_path / f"seed_{seed}" / f"{algo_name}_csv" / "se.csv")['Value'], window_size= 200)[length:])  # np.array with (5, 1000)
        Qoe_embbs.append(moving_average(pd.read_csv(csv_root_path / f"seed_{seed}" / f"{algo_name}_csv" / "qoe_embb_general.csv")['Value'], window_size= 200)[length:])  # np.array with (5, 1000)
        Qoe_urllcs.append(moving_average(pd.read_csv(csv_root_path / f"seed_{seed}" / f"{algo_name}_csv" / "qoe_urllc.csv")['Value'], window_size= 200)[length:])  # np.array with (5, 1000)
        Qoe_volte.append(moving_average(pd.read_csv(csv_root_path / f"seed_{seed}" / f"{algo_name}_csv" / "qoe_volte.csv")['Value'], window_size= 200)[length:])  # np.array with (5, 1000)
    
    results[algo_name] = compute_method_stats(
        Utilities,
        SEs,
        Qoe_embbs,
        Qoe_urllcs,
        Qoe_volte
    )
    print(f"{algo_name}")
    print(f"Utility : mean = {results[algo_name]["Utility"][0]:.2f}, std = {results[algo_name]["Utility"][1]:.2f}\n")
    print(f"SE : mean = {results[algo_name]["SE"][0]:.2f}, std = {results[algo_name]["SE"][1]:.2f}\n")
    print(f"eMBB SSR : mean = {results[algo_name]["eMBB SSR"][0]:.2f}, std = {results[algo_name]["eMBB SSR"][1]:.2f}\n")
    print(f"URLLC SSR : mean = {results[algo_name]["URLLC SSR"][0]:.2f}, std = {results[algo_name]["URLLC SSR"][1]:.2f}\n")
    print(f"VoLTE SSR : mean = {results[algo_name]["VoLTE SSR"][0]:.2f}, std = {results[algo_name]["VoLTE SSR"][1]:.2f}")
    print("=====================================\n")


