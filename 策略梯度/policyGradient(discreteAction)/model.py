#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#-*- coding: utf-8 -*-

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl


class Model(parl.Model):
    """ 使用全连接网络.

    参数:
        obs_dim (int): 观测空间的维度.
        act_dim (int): 动作空间的维度.
    """

    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        # 中间层 神经元个数
        hid1_size = act_dim * 10
        # 全连接层1（线性变换神经网络层）
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        # 全连接层2（线性变换神经网络层）
        self.fc2 = nn.Linear(hid1_size, act_dim)

    def forward(self, obs):
        # 使用 tanh 激活函数，将 全连接层1 输出的 神经元的值 进行非线性变换
        # 控制 每个神经元的值 范围为 [-1, 1] 之间
        out = paddle.tanh(self.fc1(obs))

        # 使用 softmax 激活函数，将 全连接层2 输出的 神经元的值 进行非线性变换
        # 控制 每个神经元的值 范围为 [0, 1] 之间，用以表示 每个 离散动作 对应的 概率
        # 每个神经元的值 加总和为 1.0
        # 其中 axis=-1，表示沿着观测（obs）的最后一维做 softmax操作
        prob = F.softmax(self.fc2(out), axis=-1)

        # 返回每个离散动作 对应的 概率
        return prob
