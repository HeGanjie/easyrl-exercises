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

# -*- coding: utf-8 -*-

# 检查版本
import gym
import parl
import paddle

assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"

import os
import gym
import numpy as np

from agent import Agent
from model import Model
from algorithm import PolicyGradient

from tqdm import tqdm
from parl.utils import logger, summary

# 学习率
LEARNING_RATE = 1e-3

# 训练回合数
train_episode = 1000


# 训练一个episode
def run_train_episode(agent, env):
    # 创建 观测列表、动作列表、回报（奖励）列表
    obs_list, action_list, reward_list = [], [], []
    # 环境 初始化
    obs = env.reset()
    while True:
        # 存储 观测 obs
        obs_list.append(obs)
        # 智能体根据算法采样一个随机动作
        action = agent.sample(obs)
        # 存储 动作 action
        action_list.append(action)

        # 与环境进行交换，进入 下一个观测 和 获得 当前 回报（奖励）
        obs, reward, done, info = env.step(action)
        # 存储 回报（奖励） reward
        reward_list.append(reward)

        # done = True 时，episode 结束
        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，总 reward 求平均
def run_evaluate_episodes(agent, env, render=False):
    # 评估回报（奖励）
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        # 初始化 episode 回报（奖励）
        episode_reward = 0
        while True:
            # 智能体根据算法选择一个最优动作
            action = agent.predict(obs)
            # 与环境进行交换，进入 下一个观测 和 获得 当前 回报（奖励）
            obs, reward, isOver, _ = env.step(action)
            # 加总 episode 累计回报（奖励）
            episode_reward += reward
            if render:
                # 进行一次渲染
                env.render()
            # isOver = True 时，episode 结束
            if isOver:
                break
        # 存储 该 episode 累计回报（奖励）
        eval_reward.append(episode_reward)
    # 返回 5 个episode，累计回报（奖励）的平均值
    return np.mean(eval_reward)


# 基于 蒙特卡罗策略梯度，计算该episode每个步骤的未来总回报（奖励）
def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


def main():
    # 检测目录底下是否有 policy_model 文件夹，没有则创建
    if not os.path.exists("./policy_model"):
        os.makedirs("./policy_model")

    # 创建 Cart Pole（推车杆）环境
    env = gym.make('CartPole-v1')

    # 查看 Cart Pole 观测空间 的 维度
    obs_dim = env.observation_space.shape[0]
    # 查看 Cart Pole 动作空间 的 维度
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    # 创建 神经网络模型（策略模型）
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    # 创建 策略梯度算法
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    # 创建 强化学习智能体
    agent = Agent(alg)

    # 加载模型并评估
    # if os.path.exists('./policy_model/model.ckpt'):
    #     agent.restore('./policy_model/model.ckpt')
    #     run_evaluate_episodes(agent, env, render=True)
    #     exit()

    # 设置 进度条
    pbar = tqdm(total=train_episode)
    # 训练 1000 个 episode（回合）
    for i in range(train_episode):
        # 训练一个episode，并返回该episode的 观测列表、动作列表、回报（奖励）列表
        obs_list, action_list, reward_list = run_train_episode(agent, env)

        # 每训练 10 个 episode，查看该 episode 累计回报（奖励）
        if i % 10 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        # 记录每个 episode 的 累计回报（奖励）
        summary.add_scalar('{}/training_rewards'.format('Policy Gradient'),
                           sum(reward_list), i)

        # 基于 蒙特卡罗策略梯度，采用 回合更新 的机制
        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        # 计算该episode每个步骤的未来总回报（奖励）
        batch_reward = calc_reward_to_go(reward_list)

        # 智能体进行策略学习（回合更新）
        agent.learn(batch_obs, batch_action, batch_reward)
        # 每 100 个 episode（回合），对 智能体 进行 评估
        if (i + 1) % 100 == 0:
            # render=True 查看显示效果
            # 查看 智能体 跑 5 个episode，累计回报（奖励）的平均值
            total_reward = run_evaluate_episodes(agent, env, render=False)
            logger.info('Test reward: {}'.format(total_reward))
            # 记录测试过程中，每个episode的累积回报（奖励）
            summary.add_scalar('{}/testing_rewards'.format('Policy Gradient'),
                               total_reward, i)

        # 更新进度条
        pbar.update(1)

    # save the parameters to ./policy_model/model.ckpt
    # 保存 神经网络模型（策略）参数
    agent.save('./policy_model/model.ckpt')


if __name__ == '__main__':
    main()
