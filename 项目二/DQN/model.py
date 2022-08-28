#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl


class Model(parl.Model):
    """ 使用全连接网络（线性网络）

    Args:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        # 中间层 神经元个数
        hid1_size = 128
        # 中间层 神经元个数
        hid2_size = 128
        # 全连接层1（线性变换神经网络层）
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        # 全连接层2（线性变换神经网络层）
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        # 全连接层3（线性变换神经网络层）
        self.fc3 = nn.Linear(hid2_size, act_dim)

    # Q = [ Q（obs,a1), Q(obs,a2), .... Q(obs,an) ]
    def forward(self, obs):
        # 使用 relu 激活函数，将 全连接层 输出的 神经元的值 进行非线性变换
        # 控制 每个神经元的值 如果 输入值大于0，直接返回作为输入提供的值
        # 如果 输入0或者更小，返回值 0
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        Q = self.fc3(h2)
        # Q 代表在 obs 下的 每个可选择动作 的 累积期望价值
        # 作为 Q_pi(obs_t, a_t) 的替代值的方案
        return Q
