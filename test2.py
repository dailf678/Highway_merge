import numpy as np
import rl_utils
import torch
import torch.nn as nn
from utils import get_linear_fn

# 读取/home/lifei/HighwayEnv/highway_dqn/return_list.npy
# return_list = np.load("D:\python\Highway_merge\highway_dqn\\return_list.npy")
return_list = np.load("D:\python\Highway_merge\save_models\\return_list.npy")
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
