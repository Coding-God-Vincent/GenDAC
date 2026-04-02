# -*- coding: utf-8 -*-

import itertools
import numpy as np


def action_space(total, ser_num):
    tmp = list(itertools.product(range(total + 1), repeat=ser_num))
    result = []
    for value in tmp:
        if sum(value) == total:
            result.append(list(value))
    result = np.array(result)
    [i, j] = np.where(result == 0)
    result = np.delete(result, i, axis=0)
    # print(result.shape)
    return result


def gen_state_(pkt_nums, pos):
    mean = np.array([218.8, 5338, 293])
    std = np.array([51, 847, 42.5])
    state = np.hstack(((pkt_nums - mean) / std, pos))
    return state

# 1:2:3
# 對狀態做標準化
def gen_state(pkt_nums):
    mean = np.array([218.8, 5338, 293])
    std = np.array([51, 847, 42.5])
    state = (pkt_nums - mean) / std
    return state

# 原本的 reward function
def calc__reward(qoe, se, qoe_weights= [1, 1, 1], se_weight= 0.01):
    utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]  # shape (1)
    if qoe[1] >= 0.98 and qoe[0] >= 0.98:
        if qoe[2] >= 0.95:
            if se[0] < 280:
                reward = 4
            else:
                reward = 4 + (se[0] - 280) * 0.1
        else:
            reward = (qoe[2] - 0.7) * 10
    else:
        reward = -5
    reward = np.array([reward])
    return utility, reward

# GenDAC 的 reward function
# reward : shape (1), utility.shape (1)
# se : np.int with shape (1), qoe : np.array with shape (3)
# def calc__reward(qoe, se, qoe_weights= [1, 1, 1], se_weight= 0.01, reward_clipping= False):
#     standard = 0.98  # standard for embb & volte
#     standard2 = 0.95  # standard for urllc (最高可以 0.96)
#     utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]  # shape (1)
#     if qoe[1] >= standard and qoe[0] >= standard:
#         if qoe[2] >= standard2:
#             reward = (np.matmul(qoe_weights, qoe.reshape((3, 1))) + (se_weight / 100.0) * se[0])[0] / 10  # 會介於 0~1
#         else:
#             reward = (qoe[2] - standard2) - 0.5  # -0.5~-1.45
#     else:
#         reward = -1.5  - max(0, standard - qoe[0]) - max(0, standard - qoe[1])
#     reward = np.array([reward])
#     return utility, reward

