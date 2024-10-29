import numpy as np
import random
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
from scipy.io import loadmat  # 用于加载.mat文件


# 设置字体为 SimHei
rcParams['font.family'] = 'SimHei'  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # 在这里进行池化操作
        x = torch.mean(x, dim=0, keepdim=True)  # 对第二维进行池化

        x = torch.sigmoid(self.fc3(x))  # 确保输出在[0,1]范围内
        return x.view(-1)  # 将输出展平为一维


class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样经验"""
    def __init__(self, capacity=500):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class LSHADE_Q:
    def __init__(self, fitness, constraints, lower, upper, pop_size, dim, epochs,
                 gamma=0.99, learning_rate=0.01, epsilon=0.3, min_epsilon=0.01, epsilon_decay=0.995, buffer_size=500, batch_size=32):
        self.fitness = fitness
        self.constraints = constraints
        self.lowbound = lower
        self.upbound = upper
        self.pop_size = pop_size
        self.dim = dim
        self.epochs = epochs
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # 初始化种群
        self.population = np.random.uniform(self.lowbound, self.upbound, (self.pop_size, self.dim))
        self.fit = np.array([self.fitness(ind) for ind in self.population])
        self.conv = np.array([self.constraints(ind) for ind in self.population])

        # 初始化DQN
        self.dqn = DQN(input_dim=self.dim, output_dim=2).to(self.get_device())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber损失函数
        self.model_path = 'dqn_model.pth'

        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # 初始化Pareto前沿
        self.update_pareto_front()

        # 加载真实帕累托前沿
        self.load_true_pareto_front("true_pareto_front.mat")

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_pareto_front(self):
        is_dominated = np.zeros(self.pop_size, dtype=bool)
        for i in range(self.pop_size):
            if is_dominated[i]:
                continue
            for j in range(self.pop_size):
                if i == j:
                    continue
                if np.all(self.fit[j] <= self.fit[i]) and np.any(self.fit[j] < self.fit[i]):
                    is_dominated[i] = True
                    break
        self.pareto_front = self.fit[~is_dominated]

    def load_true_pareto_front(self, file_path):
        if os.path.exists(file_path):
            try:
                data = loadmat(file_path)
                self.true_pareto_front = data['PF']
                if self.true_pareto_front.shape[0] == 2:
                    self.true_pareto_front = self.true_pareto_front.T
                print(f"成功加载真实帕累托前沿，共 {self.true_pareto_front.shape[0]} 个点。")
            except Exception as e:
                print(f"加载真实帕累托前沿时发生错误: {e}")
                self.true_pareto_front = np.empty((0, 2))
        else:
            print(f"真实帕累托前沿文件不存在: {file_path}")
            self.true_pareto_front = np.empty((0, 2))

    def mutate(self, i, F):
        idxs = np.random.choice([idx for idx in range(self.pop_size) if idx != i], 2, replace=False)
        mutant = self.population[idxs[0]] + F * (self.population[idxs[1]] - self.population[i])
        mutant = np.clip(mutant, self.lowbound, self.upbound)
        return mutant

    def crossover(self, target, mutant, CR):
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, i, trial, new_fit, new_conv):
        if self.conv[i] == 0 and new_conv == 0:
            if np.all(new_fit <= self.fit[i]) and np.any(new_fit < self.fit[i]):
                self.population[i] = trial
                self.fit[i] = new_fit
                self.conv[i] = new_conv
                return True
        elif self.conv[i] > new_conv:
            self.population[i] = trial
            self.fit[i] = new_fit
            self.conv[i] = new_conv
            return True
        return False

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # 随机选择F和CR，确保在合理范围内
            action = np.random.uniform(0, 1, 2)
            action[0] = np.clip(action[0], 0, 0.1)  # F在[0, 0.1]范围内
            action[1] = np.clip(action[1], 0, 0.5)  # CR在[0, 0.5]范围内
            return torch.tensor(action, dtype=torch.float32).to(self.get_device())
        else:
            with torch.no_grad():
                action = self.dqn(state)
                action[0] = torch.clamp(action[0], 0, 0.1)  # F在[0, 0.1]范围内
                action[1] = torch.clamp(action[1], 0, 0.5)  # CR在[0, 0.5]范围内
                print(action)
            return action

    def calculate_diversity_reward(self, population):
        # 计算种群中解之间的距离
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(population[i] - population[j])  # 计算欧几里得距离
                distances.append(distance)

        # 计算平均距离作为多样性奖励
        if distances:
            average_distance = np.mean(distances)
            return average_distance  # 返回多样性奖励
        else:
            return 0  # 如果没有解，返回0


    def learn(self):
        """从经验回放缓冲区采样并执行学习步骤"""

        # 从经验回放缓冲区中随机采样一批数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # 将采样的数据转换为PyTorch张量，并移动到指定设备（如GPU）
        states = torch.tensor(states, dtype=torch.float32).to(self.get_device())  # 当前状态
        actions = torch.tensor(actions, dtype=torch.float32).to(self.get_device())  # 采取的动作
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(
            self.get_device())  # 奖励，增加一个维度以匹配目标Q值的形状
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.get_device())  # 下一个状态
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.get_device())  # 终止标志，增加一个维度
        # 计算当前状态的Q值
        current_Q = self.dqn(states)  # 形状应为 (batch_size, output_dim)

        # 计算目标Q值
        with torch.no_grad():  # 在计算目标Q值时不需要计算梯度
            target_Q = self.dqn(next_states)  # 计算下一个状态的Q值，形状应为 (batch_size, output_dim)
            max_target_Q, _ = torch.max(target_Q, dim=1, keepdim=True)  # 获取下一个状态的最大Q值，形状为 (batch_size, 1)
            target = rewards + self.gamma * max_target_Q * (1 - dones)  # 计算目标Q值，考虑折扣因子和终止状态

        # 因为动作是连续的 [F, CR]，而不是离散动作索引，
        # 我们需要修改损失计算方式。这里我们假设 DQN 输出的是对每个动作维度的 Q 值，
        # 因此我们可以直接使用当前_Q 和 target 进行回归。
        # 如果 DQN 的输出是具体的动作值，可以考虑使用均方误差损失。
        # 使用均方误差损失
        criterion = nn.MSELoss()
        loss = criterion(current_Q, target)
        # 优化模型前记录参数的范数
        param_norm_before = 0.0
        for param in self.dqn.parameters():
            param_norm_before += param.data.norm(2).item()
            # 优化模型
        self.optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 反向传播，计算当前损失的梯度
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        self.optimizer.step()  # 更新模型参数

        # 优化模型后记录参数的范数
        param_norm_after = 0.0
        for param in self.dqn.parameters():
            param_norm_after += param.data.norm(2).item()

        # 打印参数变化情况
        print( f"参数范数变化前: {param_norm_before:.4f}, 变化后: {param_norm_after:.4f}, 差异: {param_norm_after - param_norm_before:.6f}")
        return loss.item()  # 返回当前损失值

    def run(self):
        if os.path.exists(self.model_path):
            self.dqn.load_state_dict(torch.load(self.model_path, map_location=self.get_device()))
            print("加载预训练模型成功，继续训练...")
        else:
            print("预训练模型不存在，开始训练...")

        no_improve_epochs = 0
        old_diversity_reward = self.calculate_diversity_reward(self.population)  # 用于多样性分数的计算和保存
        for epoch in range(1, self.epochs + 1):
            epoch_reward = 0
            epoch_no_improve_pop = 0
            previous_fit = self.fit.copy()  # 保存上一时刻的适应度
            epoch_new_fit_sum = 0  # 初始化当前 epoch 的适应度总和
            # 选择动作，获取 F 和 CR
            # 将整个种群的信息转换为张量
            state = torch.tensor(self.population, dtype=torch.float32).to(self.get_device())
            action = self.choose_action(state)
            F, CR = action.cpu().numpy()
            for i in range(self.pop_size):
                mutant = self.mutate(i, F)
                trial = self.crossover(self.population[i], mutant, CR)
                new_fit = self.fitness(trial)
                new_conv = self.constraints(trial)
                selection = self.select(i, trial, new_fit, new_conv)

                # 添加经验到回放缓冲区
                done = False  # ZDT1 无终止条件
                reward = self.calculate_diversity_reward(self.population)  # 使用多样性奖励
                self.replay_buffer.push(self.population[i], action.cpu().numpy(), reward, self.population[i], done)

                # 更新帕累托前沿
            self.update_pareto_front()

            # 计算多样性奖励的变化
            new_diversity_reward = self.calculate_diversity_reward(self.population)
            epoch_reward = new_diversity_reward - old_diversity_reward
            old_diversity_reward = new_diversity_reward  # 更新旧的多样性奖励

            print(f'第 {epoch} 轮, 奖励: {epoch_reward:.4f}')

            if epoch_no_improve_pop / self.pop_size > 0.8:
                no_improve_epochs += 1
            else:
                no_improve_epochs = 0

                # 衰减epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.min_epsilon)

            # 执行学习步骤
            if len(self.replay_buffer) >= self.batch_size:
                loss = self.learn()
                print(f"训练损失: {loss:.4f}")

            if no_improve_epochs > 10:
                break

            # 训练完成后保存模型
        torch.save(self.dqn.state_dict(), self.model_path)
        print("模型已保存到:", self.model_path)

    def show_result(self):
        """展示帕累托前沿和种群的适应度"""
        plt.figure(figsize=(10, 6))

        # 绘制真实帕累托前沿
        if self.true_pareto_front.size > 0:
            plt.scatter(self.true_pareto_front[:, 0], self.true_pareto_front[:, 1],
                        color='green', label='真实 Pareto 前沿', marker='x')

        # 绘制帕累托前沿
        if self.pareto_front.size > 0 and self.pareto_front.ndim == 2:
            plt.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1],
                        color='blue', label='获得的 Pareto 前沿', alpha=0.6)        # 绘制所有个体的适应度
        if self.fit.ndim == 2:
            plt.scatter(self.fit[:, 0], self.fit[:, 1],
                        color='red', label='所有个体', alpha=0.3)
        # 设置图形的标签和标题
        plt.xlabel('目标 1 (f1)')
        plt.ylabel('目标 2 (f2)')
        plt.title('Pareto 前沿和种群适应度')
        plt.legend()
        plt.grid(True)
        plt.show()



# 定义ZDT1多目标适应度函数
def fitness(x):
    """计算ZDT1的多目标适应度值"""
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - sqrt(f1 / g))
    return np.array([f1, f2])

# 定义约束函数（ZDT1无约束）
def constraints(x):
    """计算约束违反程度，ZDT1无约束返回0"""
    return 0.0


if __name__ == "__main__":
    # 初始化LSHADE_Q实例并运行
    lshade_q = LSHADE_Q(
        fitness=fitness,  # 适应度函数
        constraints=constraints,  # 约束条件
        lower=0.0,  # ZDT1定义域下限，通常为0
        upper=1.0,  # ZDT1定义域上限，通常为1
        pop_size=150,  # 种群规模，减少以适应低配置电脑
        dim=10,  # 设置维度为10
        epochs=200,  # 训练轮数，减少以加快训练速度
        gamma=0.99,  # 折扣因子，用于未来奖励的权重
        learning_rate=0.001,  # 学习率，控制模型更新的步长
        epsilon=0.1,  # 初始探索率，用于平衡探索与利用
        min_epsilon=0.01,  # 最小探索率，防止过度利用
        epsilon_decay=0.995,  # 探索率衰减因子，控制探索率的降低速度
        buffer_size=500,  # 经验回放缓冲区大小，存储过去的经验
        batch_size=16  # 批量大小，控制每次训练使用的样本数量
    )

    lshade_q.run()
    lshade_q.show_result()



