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

import copy
import parl
import paddle
from parl.utils.utils import check_model_method


class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): 表示 Q 函数的前向神经网络.
            gamma (float): 计算累积奖励的折扣因子
            lr (float): 学习率.
        """
        # checks
        check_model_method(model, 'forward', self.__class__.__name__)
        # 判断gamma折扣因子的数据类型为 float
        assert isinstance(gamma, float)
        # 判断lr学习率的数据类型为 float
        assert isinstance(lr, float)

        # 神经网络模型（Q函数）
        self.model = model
        # 目标神经网络模型（Target Q函数） ---> 固定Q目标
        self.target_model = copy.deepcopy(model)

        # 折扣因子
        self.gamma = gamma
        # 学习率
        self.lr = lr

        # 均方差误差损失函数
        self.mse_loss = paddle.nn.MSELoss(reduction='mean')

        # Adam 神经网络参数优化器（梯度下降法）
        self.optimizer = paddle.optimizer.Adam(
            # 学习率
            learning_rate=lr,
            # 指定优化器需要优化的参数
            parameters=self.model.parameters())

    # 根据输入观测，预测输出动作价值
    def predict(self, obs):
        """ 使用 self.model（Q 函数）来预测动作值
        """
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用 DQN 算法更新 Q 函数（self.model）
        """
        # 预测观测obs下的离散动作的价值 Q(obs,a)
        pred_values = self.model(obs)
        # 动作空间的维度
        action_dim = pred_values.shape[-1]
        # paddle.squeeze将输入Tensor的Shape中尺寸为1的维度进行压缩
        # 返回对维度进行压缩后的Tensor，数据类型与输入Tensor一致
        # axis 代表要压缩的轴，默认为None，表示对所有尺寸为1的维度进行压缩。
        # 这边 输入 axis=-1，表示对最后一维数据维度为1的话，将进行压缩。
        action = paddle.squeeze(action, axis=-1)
        # 将动作action 转换为 独热编码（one-hot encoding）
        # 其中 num_classes 用于定义一个one-hot向量的长度
        action_onehot = paddle.nn.functional.one_hot(
            action, num_classes=action_dim)
        # 动作价值Q(obs,a) = 动作价值（神经网络模型 Q函数） * 动作的独热编码（one-hot encoding）
        pred_value = pred_values * action_onehot
        # 求和运算sum的维度 axis=1，代表第一维；keepdim=True，代表在输出Tensor中保留减小的维度
        pred_value = paddle.sum(pred_value, axis=1, keepdim=True)

        # 目标神经网络模型（Target Q函数）
        # 创建一个上下文来禁用动态图梯度计算。在此模式下，每次计算的结果都将具有 stop_gradient=True
        with paddle.no_grad():
            # max_v = maxQ(obs_{t+1}, a_{t+1})
            # 站在 obs_{t+1} 历史数据基础上，策略默认选择当前Q值最大的动作 a_{t+1}
            max_v = self.target_model(next_obs).max(1, keepdim=True)
            # 目标函数为 r + gamma * max_v
            # 当 terminal=1 代表回合结束，在最后一个状态，此时目标函数为 r(当前奖励)
            target = reward + (1 - terminal) * self.gamma * max_v
        # 找到 神经网络参数化向量 最小化 值函数近似值(pred_value) 和 目标值 之间的均方误差
        loss = self.mse_loss(pred_value, target)

        # 清空梯度
        self.optimizer.clear_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        self.optimizer.step()
        return loss

    def sync_target(self):
        """ 将训练网络的参数分配给目标网络
        """
        self.model.sync_weights_to(self.target_model)
