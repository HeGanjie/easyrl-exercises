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

import parl
import paddle
import numpy as np


class Agent(parl.Agent):
    def __init__(self, algorithm):
        super(Agent, self).__init__(algorithm)

    # 根据输入观测，采样输出的离散动作，带探索
    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        # 将 观测obs 转换为 Tensor张量
        obs = paddle.to_tensor(obs, dtype='float32')
        # 根据算法，返回每个离散动作选择的概率
        prob = self.alg.predict(obs)
        # 将 动作概率prob 转换为 Numpy数组
        prob = prob.numpy()

        # 根据动作概率选取动作
        # 从 离散动作中，根据 每个离散动作的动作概率prob，随机选取 1 个动作
        act = np.random.choice(len(prob), 1, p=prob)[0]
        return act

    # 根据输入观测，预测输出最优的离散动作
    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        # 将 观测obs 转换为 Tensor张量
        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        # 根据动作概率选择概率最高的动作
        # 从 离散动作中，根据 每个离散动作的动作概率prob，选取概率最高的 1个 最优动作
        act = prob.argmax().numpy()[0]
        return act

    def learn(self, obs, act, reward):
        """ 根据训练数据更新一次模型参数
        """
        # act为动作数组，添加维度 ---> [act]
        act = np.expand_dims(act, axis=-1)
        # 回报（奖励）数组，添加维度 ---> [reward]
        reward = np.expand_dims(reward, axis=-1)

        # 将 观测obs数组 转换为 Tensor张量
        obs = paddle.to_tensor(obs, dtype='float32')
        # 将 动作act数组 转换为 Tensor张量，离散动作，以 整数 表示
        act = paddle.to_tensor(act, dtype='int32')
        # 将 回报（奖励）reward数组 转换为 Tensor张量
        reward = paddle.to_tensor(reward, dtype='float32')

        # 根据算法返回损失函数值
        loss = self.alg.learn(obs, act, reward)
        return loss.numpy()[0]
