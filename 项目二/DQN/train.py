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

import os
import gym
assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"

import numpy as np
import parl
import paddle

import argparse
from tqdm import tqdm
from parl.utils import logger, summary, ReplayMemory

from agent import Agent
from model import Model
from algorithm import DQN

# 训练间隔，每隔 5 步训练一次模型
LEARN_FREQ = 5  # training frequency
# 经验池大小
MEMORY_SIZE = 200000
# 经验池预存数据大小
MEMORY_WARMUP_SIZE = 200
# 用于训练学习价值函数的，数据批量
BATCH_SIZE = 64
# 学习率
LEARNING_RATE = 0.0005
# 折扣因子
GAMMA = 0.99


# train an episode
def run_train_episode(agent, env, rpm):
    # 回合累积奖励
    total_reward = 0
    # 初始化环境
    obs = env.reset()
    # 回合步数
    steps = 0
    while True:
        steps += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        # 将 (obs_t, a_t, r_t, obs_{t+1}) 存入 经验池
        rpm.append(obs, action, reward, next_obs, done)

        # train model
        # 训练Q函数模型
        # 经验池数据量 大于 预存数据量  and  当前步数是 训练间隔步数 的倍数
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (steps % LEARN_FREQ == 0):
            # 从经验池中 均匀 抽取 BATCH_SIZE 大小 的 历史数据
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample_batch(BATCH_SIZE)
            # 训练Q函数模型，并返回 损失函数
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward, steps


# 评估 agent, 跑 5 个episode，总 reward 求平均
def run_evaluate_episodes(agent, env, eval_episodes=5, render=False):
    eval_reward = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    # 检测目录底下是否有 Qfunction_model 文件夹，没有则创建
    if not os.path.exists("./Qfunction_model"):
        os.makedirs("./Qfunction_model")

    # 创建 Cart Pole（推车杆）环境
    env = gym.make('CartPole-v1')

    # 查看 Cart Pole 观测空间 的 维度
    obs_dim = env.observation_space.shape[0]
    # 查看 Cart Pole 观测空间 的 维度
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 在离散控制环境中设置 action_shape = 0
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)

    # 根据parl框架构建agent
    # 创建 神经网络模型（Q函数模型）
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    # 创建 深度Q学习算法
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    # 创建 深度强化学习智能体
    agent = Agent(
        alg, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-6)

    # 加载模型并评估
    ''' 
    if os.path.exists('./Qfunction_model/model.ckpt'):
        agent.restore('./Qfunction_model/model.ckpt')
        run_evaluate_episodes(agent, env, render=True)
        exit()
    '''

    # 填充预先数据到 经验池
    with tqdm(
            total=MEMORY_WARMUP_SIZE, desc='[Replay Memory Warm Up]') as pbar:
        while len(rpm) < MEMORY_WARMUP_SIZE:
            total_reward, steps = run_train_episode(agent, env, rpm)
            pbar.update(steps)

    # 最大回合数
    max_episode = args.max_episode

    # 开始训练
    pbar = tqdm(total=max_episode)
    episode = 0
    while episode < max_episode:
        # train part
        for i in range(50):
            total_reward, steps = run_train_episode(agent, env, rpm)
            logger.info('episode:{}    Train reward:{}'.format(
                episode, total_reward))
            # 记录每个 episode 的 累计回报（奖励）
            summary.add_scalar('{}/training_rewards'.format('DQN'),
                               total_reward, episode)
            episode += 1

        pbar.update(50)
        # test part
        eval_reward = run_evaluate_episodes(agent, env, render=False)
        logger.info('episode:{}    e_greed:{}   Test reward:{}'.format(
            episode, agent.e_greed, eval_reward))
        summary.add_scalar('{}/testing_rewards'.format('DQN'),
                           eval_reward, episode)


    # save the parameters to ./Qfunction_model/model.ckpt
    save_path = './Qfunction_model/model.ckpt'
    agent.save(save_path)

    '''
    保存策略网络的模型和参数以进行推理
    save_inference_path = './inference_model'
    input_shapes = [[None, env.observation_space.shape[0]]]
    input_dtypes = ['float32']
    agent.save_inference_model(save_inference_path, input_shapes, input_dtypes)
    '''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_episode',
        type=int,
        default=1600,
        help='stop condition: number of max episode')
    args = parser.parse_args()

    main()
