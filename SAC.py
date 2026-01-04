import torch.nn as nn
import torch
import numpy as np
from tqdm.auto import auto
from pathlib import Path
import matplotlib.pyplot as plt
from Env.env_fixedUE import cellularEnv
from Env.env_movingUE import EnvMove

from Utils.SAC_utils import Model, ReplayBuffer, SACopt

'''SAC 作法硬傷
    若直接使用 SAC 那一套，只不過因為這邊是分配資源場景所以我在 tanh 輸出後直接硬套上一個 Softmax，讓其總合為 1。
    這樣做最簡單好改但也會有一個硬傷 : 就是他沒辦法去嘗試很極端的資源分配。
    ex : 假設經過 tanh 之後輸出 : [1, -1, -1] (這已經是最極端的分配的)，那這時候經過 softmax 後會輸出 : [0.787, 0.106, 0.106]
    要解決這個問題最常用的做法就是對 tanh 輸出後的數值最一個放大 : 同乘一個 tau，這邊建議設 5，因為這樣就能夠達到很極端的配置了。
    ex : 假設經過 tanh 之後輸出 : [1, -1, -1] (這已經是最極端的分配的)，乘上 5 之後就變成 [5, -5, -5] 會得到 [0.9999..., 4.5396e-05, 4.5396e-05]
'''







