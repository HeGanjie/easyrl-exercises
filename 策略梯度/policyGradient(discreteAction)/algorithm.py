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
import paddle.nn.functional as F


class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr):
        """ Policy Gradient algorithm
        
        Args:
            model (parl.Model): policy的前向网络.
            lr (float): 学习率.
        """
        # 判断lr学习率的数据类型为 float
        assert isinstance(lr, float)

        # 神经网络模型（策略）
        self.model = model

        # Adam 神经网络参数优化器（梯度下降法）
        self.optimizer = paddle.optimizer.Adam(
            # 学习率
            learning_rate=lr,
            # 指定优化器需要优化的参数
            parameters=self.model.parameters()
        )

    # 根据输入观测，预测输出每个离散动作的概率
    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        prob = self.model(obs)
        # 在多个离散动作的情况下，返回每个离散动作选择的概率
        return prob

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        # 输入 观测obs张量（Tensor）到 神经网络模型（策略）
        # 获取神经网络（策略）输出的 动作概率prob张量（Tensor）
        prob = self.model(obs)

        # paddle.squeeze将输入Tensor的Shape中尺寸为1的维度进行压缩
        # 返回对维度进行压缩后的Tensor，数据类型与输入Tensor一致
        action_onehot = paddle.squeeze(

            # 将动作action 转换为 独热编码（one-hot encoding）
            # 其中 num_classes 用于定义一个one-hot向量的长度
            F.one_hot(action, num_classes=prob.shape[1]),

            # axis 代表要压缩的轴，默认为None，表示对所有尺寸为1的维度进行压缩。
            # 这边 输入 axis=1，表示对第1维数据维度为1的话，将进行压缩。
            axis=1)

        # 动作对数概率 = sum（对数动作概率（同策略神经网络输出的） * 动作的独热编码（one-hot encoding））
        # 求和运算sum的维度 axis=-1，代表最后一维
        log_prob = paddle.sum(paddle.log(prob) * action_onehot, axis=-1)

        # 对 reward张量（Tensor）第1维数据维度为1的话，将进行压缩。
        reward = paddle.squeeze(reward, axis=1)

        # 对 每个动作对数概率 * 对应动作的未来累计总回报（奖励）
        # 目标是 训练 使得 智能体的策略 能够 实现 每一步动作的未来累计总回报（奖励）最大化
        # 而 Adam 神经网络参数优化器 是 梯度下降
        # (-1 * ) 操作来实现 梯度上升
        loss = paddle.mean(-1 * log_prob * reward)

        # 清空梯度
        self.optimizer.clear_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        self.optimizer.step()
        return loss
