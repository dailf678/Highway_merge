import numpy as np
import rl_utils
import torch
import torch.nn as nn

# 读取/home/lifei/HighwayEnv/highway_dqn/return_list.npy
return_list = np.load("D:\python\HighwayEnv\save_models\\return_list.npy")
# 将return_list 画成图
import matplotlib.pyplot as plt

# plt.plot(return_list)
# plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(mv_return)
plt.show()
