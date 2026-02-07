import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Exponential Moving Average, bigger the weight (0~1) smoother the line
def ema(values, weight= 0.97):
    values = np.asarray(values, dtype= float)
    smoothed = np.zeros_like(values)  # 創建一個跟 values 一樣大的 np.zeros
    last = values[0]
    for i, v in enumerate(values):
        last = weight * last + (1 - weight) * v
        smoothed[i] = last
    return smoothed


def load_seed_series(root_dir: Path, seeds, filename= "utility.csv"):
    series = []
    for s in seeds:
        csv_path = root_dir / f"{s}_csv" / filename  # root/exp4_csv/utility.csv
        v = pd.read_csv(csv_path)["Value"].to_numpy()
        series.append(v)
    return series


def rolling_std(values, window=200):
    values = np.asarray(values, dtype=float)
    return pd.Series(values).rolling(window=window, min_periods=window).std().to_numpy()[200:]


def stack_rolling_std(runs):
    # 對每個 seed 算 rolling std，並對齊長度（取最短者避免長度不同）
    rs_list = [rolling_std(v, window= ROLLING_WINDOW) for v in runs]
    L = min(len(x) for x in rs_list)
    rs = np.vstack([x[:L] for x in rs_list])  # shape = (num_seeds, L)
    return rs


def stability_score(values, window= 200, start_ep= None):
    rs = rolling_std(values, window)
    if start_ep is not None:
        rs = rs[start_ep:]
    return np.nanmean(rs)




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''fixedUE 跟 movingUE 的 Hard slicing training curve
用以說明 movingUE 環境的波動。
'''

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Hyperparameters
algo_name = "Hard_Slicing"
image_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Combine/MovingUE_fluctuation")
# load utilities
fixed_exps = ['exp4', 'exp5', 'exp6', 'exp7', 'exp8']
moving_exps = ['exp3', 'exp4', 'exp5', 'exp6', 'exp7']

ROLLING_WINDOW = 200
START_EP = 2000

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'utility'

fixed_csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/Hard_Slicing_TC")
moving_csv_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/Hard_Slicing_TC")

fixed_utility_series = load_seed_series(root_dir= fixed_csv_path, seeds= fixed_exps)
moving_utility_series = load_seed_series(root_dir= moving_csv_path, seeds= moving_exps)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# plot figure
'''1. Rolling std'''

fixed_rs  = stack_rolling_std(fixed_utility_series)
moving_rs = stack_rolling_std(moving_utility_series)


x = np.arange(fixed_rs.shape[1])

fixed_mean, fixed_std   = np.nanmean(fixed_rs, axis=0), np.nanstd(fixed_rs, axis=0)
moving_mean, moving_std = np.nanmean(moving_rs, axis=0), np.nanstd(moving_rs, axis=0)

plt.figure(figsize=(6,4))
plt.title("Rolling std of Utility (Hard Slicing) - mean ± std across seeds")
plt.xlabel("Episode")
plt.ylabel("Rolling std")
# fixedUE
plt.plot(x, fixed_mean,  color="tab:green", label="fixed_UE")
plt.fill_between(x, fixed_mean-fixed_std, fixed_mean+fixed_std, color="tab:green", alpha=0.2)
# movingUE
plt.plot(x, moving_mean, color="tab:red",   label="moving_UE")
plt.fill_between(x, moving_mean-moving_std, moving_mean+moving_std, color="tab:red", alpha=0.2)

plt.legend(
    fontsize= 'small',  # 字體大小
    labelspacing= 0.2,  # 垂直標籤之間的間距
    handletextpad= 0.5,  # 圖示與文字之間的間距
    borderaxespad= 0.5,  # 圖例框與邊框的間距
    # ncol= 1  # 2 : 橫向、1 : 垂直
)
plt.savefig(image_path / "Rolling_std")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''2. Stability score (bar)'''

fixed_scores  = [stability_score(v, window=ROLLING_WINDOW, start_ep=START_EP) for v in fixed_utility_series]
moving_scores  = [stability_score(v, window=ROLLING_WINDOW, start_ep=START_EP) for v in moving_utility_series]

# plt.figure(2)
# plt.clf()
# plt.title("Stability Score (Hard Slicing) across seeds")
# plt.ylabel("Mean Rolling std")
# plt.boxplot(
#     [fixed_scores, moving_scores],
#     labels=["fixed_UE", "moving_UE"],
#     showfliers=True
# )

plt.figure(figsize=(4,4))
plt.title("Stability Score (Hard Slicing)")

positions = [1, 2]

plt.boxplot(
    [fixed_scores, moving_scores],
    positions=positions,
    widths=0.4,
    showfliers=False
)

# jitter scatter
for i, scores in enumerate([fixed_scores, moving_scores]):
    x = np.random.normal(positions[i], 0.04, size=len(scores))
    plt.scatter(x, scores, alpha=0.8)

plt.xticks(positions, ["fixed_UE", "moving_UE"])
plt.ylabel("Mean Rolling Std")
plt.tight_layout()


# plt.legend(
#     fontsize= 'small',  # 字體大小
#     labelspacing= 0.2,  # 垂直標籤之間的間距
#     handletextpad= 0.5,  # 圖示與文字之間的間距
#     borderaxespad= 0.5,  # 圖例框與邊框的間距
#     # ncol= 1  # 2 : 橫向、1 : 垂直
# )
plt.tight_layout()
plt.savefig(image_path / "Stability_score")









