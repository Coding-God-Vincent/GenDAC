import numpy as np
import pandas as pd
from pathlib import Path

# ===== EMA =====
def ema(values, weight=0.9):
    values = np.asarray(values, dtype=float)
    smoothed = np.zeros_like(values)
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

# ===== Feasible Ratio =====
def compute_feasible_ratio(eMBB, URLLC, VoLTE, threshold=0.95):
    """
    eMBB, URLLC, VoLTE: shape (num_seeds, T)
    return:
        mean_ratio: across seeds mean feasible ratio
        std_ratio: across seeds std feasible ratio
    """
    eMBB = np.asarray(eMBB, dtype=float)
    URLLC = np.asarray(URLLC, dtype=float)
    VoLTE = np.asarray(VoLTE, dtype=float)

    ratios = []

    for i in range(eMBB.shape[0]):
        feasible = (
            (eMBB[i] >= threshold) &
            (URLLC[i] >= threshold) &
            (VoLTE[i] >= threshold)
        )
        ratio = np.sum(feasible) / feasible.size
        ratios.append(ratio)

    ratios = np.asarray(ratios, dtype=float)

    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios, ddof=1)

    return mean_ratio, std_ratio


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''hyperparameters'''
fixed = False
seeds = [124, 125, 126, 127, 128]
# seeds = [124]
threshold = 0.95

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

csv_root_path = Path("Outcome_github/CSVs")


# algo_names = ["GenDAC", "GANDDQN", "LSTM_A2C", "Hard_Slicing", "PPO", "SAC"]
algo_names = ["GenDAC_bs_1", "GenDAC", "GenDAC_bs_5"]


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
results = {}

for algo_name in algo_names:

    # 暫存五個 seeds 的完整 SSR 內容
    Qoe_embbs_all = []
    Qoe_urllcs_all = []
    Qoe_volte_all = []

    for seed in seeds:
        embb_vals = pd.read_csv(
            csv_root_path / f"seed_{seed}" / f"{algo_name}_csv" / "qoe_embb_general.csv"
        )["Value"].to_numpy()

        urllc_vals = pd.read_csv(
            csv_root_path / f"seed_{seed}" / f"{algo_name}_csv" / "qoe_urllc.csv"
        )["Value"].to_numpy()

        volte_vals = pd.read_csv(
            csv_root_path / f"seed_{seed}" / f"{algo_name}_csv" / "qoe_volte.csv"
        )["Value"].to_numpy()

        # Feasible Ratio 建議用 raw values，不要 EMA
        Qoe_embbs_all.append(embb_vals)
        Qoe_urllcs_all.append(urllc_vals)
        Qoe_volte_all.append(volte_vals)

    mean_ratio, std_ratio = compute_feasible_ratio(
        Qoe_embbs_all,
        Qoe_urllcs_all,
        Qoe_volte_all,
        threshold=threshold
    )

    results[algo_name] = {
        "Feasible Ratio": (mean_ratio, std_ratio)
    }

    print(f"{algo_name}")
    print(f"Feasible Ratio (SSR >= {threshold} for all slices): mean = {mean_ratio:.2f}, std = {std_ratio:.2f}")
    print("=====================================\n")