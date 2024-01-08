import numpy as np
import rl_utils
import torch
import torch.nn as nn
from utils import get_linear_fn

# 读取/home/lifei/HighwayEnv/highway_dqn/return_list.npy
return_list = np.load("D:\python\Highway_merge\merge_ppo\\return_list.npy")
# return_list = np.load("D:\python\Highway_merge\save_models\\return_list.npy")
# 将return_list 画成图
import matplotlib.pyplot as plt

plt.plot(return_list)
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(mv_return)
plt.show()
# exploration_schedule = get_linear_fn(
#     1,
#     0.05,
#     0.1,
# )
# exploration_rate = exploration_schedule(200 / 2000)
# print(exploration_rate)


# def Distance_brake(speed, a):
#     return speed**2 / (2 * a)


# def Distance_free(speed, t):
#     return speed * t


# x_lead = 130
# vx_lead = 20
# # get next row
# x_lag = 115
# vx_lag = 18
# d_brake_ego = Distance_brake(20, 5)
# d_free_ego = Distance_free(20, 0.3)
# d_brake_lead = Distance_brake(vx_lead, 5)
# d_free_lag = Distance_free(vx_lag, 0.3)
# d_brake_lag = Distance_brake(vx_lag, 5)
# print(d_brake_ego)
# print(d_free_ego)
# print(d_brake_lead)
# print(d_free_lag)
# print(d_brake_lag)
# d_safe_lead = (d_free_ego + d_brake_ego) - d_brake_lead
# d_safe_lag = (d_free_lag + d_brake_lag) - d_brake_ego
# g_safe = max(d_safe_lead, d_safe_lag)
# g_real = x_lead - x_lag
# print(d_safe_lead)
# print(d_safe_lag)
# print(g_safe)
# print(g_real)
