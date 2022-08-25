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

import parl
import paddle
import numpy as np


class Agent(parl.Agent):
    """

    Args:
        algorithm(parl.Algorithm): algorithm used to solve the problem.

    """

    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=0):
        super(Agent, self).__init__(algorithm)
        # 判断act_dim动作维度的数据类型为 int
        assert isinstance(act_dim, int)
        self.act_dim = act_dim
        # 记录全局的步数
        self.global_step = 0
        # 目标网络更新的 间隔 = 200步 更新一次
        self.update_target_steps = 200
        # 有一个小的概率e去基于e去均匀的采样所有的action
        self.e_greed = e_greed
        # e_greed 衰减
        self.e_greed_decrement = e_greed_decrement

    # 根据输入观测，采样输出的离散动作，带探索
    def sample(self, obs):
        """给定观测时对“探索”动作进行采样

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            act(int): action
        """
        sample = np.random.random()
        # 概率小于 e 均匀的采样所有的 action
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        else:
            # 保留 0.01 的概率 来 均匀的采样所有的action
            if np.random.random() < 0.01:
                act = np.random.randint(self.act_dim)
            # 根据算法，选择 当前 观测obs 下 Q价值最大的动作
            else:
                act = self.predict(obs)
        # 概率e 衰减，随着智能体和环境动态交互的次数
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs):
        """给定观测时预测动作

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            act(int): action
        """
        # 将 观测obs 转换为 Tensor张量
        obs = paddle.to_tensor(obs, dtype='float32')
        # # 根据算法，返回在当前观测obs下，每个离散动作对应的Q值
        pred_q = self.alg.predict(obs)
        # 选择Q值最大的动作
        act = pred_q.argmax().numpy()[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        """使用回合数据更新模型，这边使用 时序差分TD方法 进行学习

        Args:
            obs(np.float32): shape of (batch_size, obs_dim)
            act(np.int32): shape of (batch_size)
            reward(np.float32): shape of (batch_size)
            next_obs(np.float32): shape of (batch_size, obs_dim)
            terminal(np.float32): shape of (batch_size)

        Returns:
            loss(float)

        """
        # 全局的步数 为 目标网络更新的间隔步数 的 倍数
        if self.global_step % self.update_target_steps == 0:
            # 更新目标网络（硬更新）
            self.alg.sync_target()
        self.global_step += 1

        # act为动作数组，添加维度 ---> [act]
        act = np.expand_dims(act, axis=-1)
        # 回报（奖励）数组，添加维度 ---> [reward]
        reward = np.expand_dims(reward, axis=-1)
        # 结束 数组，添加维度 ---> [terminal]
        terminal = np.expand_dims(terminal, axis=-1)

        # 将 观测obs数组 转换为 Tensor张量
        obs = paddle.to_tensor(obs, dtype='float32')
        # 将 动作act数组 转换为 Tensor张量，离散动作，以 整数 表示
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        # 根据算法返回损失函数值
        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss.numpy()[0]
