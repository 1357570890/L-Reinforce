import numpy as np
import random
from math import cos, pi, sin
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os

class DQN(nn.Module):  # 定义网络结构
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 使用sigmoid将输出限制在[0, 1]
        x = self.fc4(x)  # 不使用激活函数，直接输出
        return x

class LSHADE_Q:
    def __init__(self, fitness, constraints, lower, upper, pop_size, dim, epochs):
        self.fitness = fitness  # 适应度函数
        self.constraints = constraints  # 约束条件
        self.lowbound = lower  # 下界
        self.upbound = upper  # 上界
        self.pop_size = pop_size  # 种群大小
        self.dim = dim  # 维度
        self.population = np.random.rand(self.pop_size, self.dim)  # 种群
        self.fit = np.random.rand(self.pop_size)  # 适应度
        self.conv = np.random.rand(self.pop_size)  # 约束
        self.best = self.population[0]  # 最优个体
        self.Epochs = epochs  # 迭代次数
        self.NFE = 0  # 函数评价次数
        self.max_NFE = 10000 * self.dim
        self.dqn = DQN(input_dim=self.dim, output_dim=2)  # 定义DQN模型
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)  # 定义优化器
        self.model_path = 'dqn_model.pth'  # 模型文件路径
        self.best_fitness_history = []  # 存储每个epoch的最佳适应度

    def initpop(self):
        # 初始化种群
        self.population = self.lowbound + (self.upbound - self.lowbound) * np.random.rand(self.pop_size, self.dim)
        # 初始化适应度
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])
        # 初始化约束
        self.conv = np.array([self.constraints(chrom) for chrom in self.population])
        # 更新函数评价次数
        self.NFE += self.pop_size
        # 更新最优个体
        self.best = self.population[np.argmin(self.fit)]

    def mut(self, i, F):
        # 选择两个随机个体
        idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 2, replace=False)
        # 引入变异因子的随机性
        F = torch.clamp(F + np.random.uniform(-0.2, 0.2), 0, 1)  # 增加变异范围
        # 计算变异向量
        v = torch.tensor(self.population[idxs[0]], dtype=torch.float32, requires_grad=True) + F * (torch.tensor(self.population[idxs[1]], dtype=torch.float32, requires_grad=True) - torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True))
        # 边界处理
        return torch.clamp(v, self.lowbound, self.upbound)

    def cross(self, mut_chrom, i, CR):
        # 初始化交叉个体
        cross_chrom = torch.clone(torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True))
        # 选择一个随机维度
        j = np.random.randint(0, self.dim)
        # 对每个维度进行交叉
        for k in range(self.dim):
            if np.random.rand() < CR.detach().numpy() or k == j:
                # 交叉
                cross_chrom[k] = mut_chrom[k]
        return cross_chrom

    def select(self, cross_chrom, i):
        # 计算适应度
        temp = self.fitness(cross_chrom.detach().numpy())
        # 计算约束
        temp_v = self.constraints(cross_chrom.detach().numpy())
        # 更新函数评价次数
        self.NFE += 1
        # 如果约束满足并且适应度更好
        if (self.conv[i] == 0 and temp_v == 0 and self.fit[i] >= temp) or (self.conv[i] > temp_v):
            # 更新个体
            self.population[i] = cross_chrom.detach().numpy()
            # 更新适应度
            self.fit[i] = temp
            # 更新约束
            self.conv[i] = temp_v

    def run(self):
        # 检查是否存在预训练模型
        if os.path.exists(self.model_path):
            self.dqn.load_state_dict(torch.load(self.model_path))  # 加载模型
            print("加载预训练模型成功...")
        else:
            print("预训练模型不存在，开始训练...")

        # 初始化种群
        self.initpop()

        # 迭代
        for epoch in range(self.Epochs):
            # 对每个个体进行变异和交叉
            for i in range(self.pop_size):
                # 获取当前状态
                state = torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True)
                action = self.dqn(state)
                if action is None:
                    print("DQN模型输出为 None，跳过此个体。")
                    continue
                F = action[0] * (self.upbound - self.lowbound) + self.lowbound  # 缩放F值
                CR = action[1]  # CR值应在[0, 1]之间
                # 变异
                mut_chrom = self.mut(i, F)
                # 交叉
                cross_chrom = self.cross(mut_chrom, i, CR)
                # 选择
                self.select(cross_chrom, i)

                # 计算奖励
                reward = torch.tensor(self.fitness(cross_chrom.detach().numpy()) - self.fitness(self.population[i]),
                                      dtype=torch.float32, requires_grad=True)
                # 计算损失
                loss = -reward
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 更新最优个体
            self.best = self.population[np.argmin(self.fit)]
            # 存储最佳适应度
            self.best_fitness_history.append(np.min(self.fit))
            # 打印信息
            print(f'Epoch: {epoch}, Best Fitness: {np.min(self.fit)}, Best Individual: {self.best}')

        # 训练完成后保存模型
        torch.save(self.dqn.state_dict(), self.model_path)
        print("模型已保存到:", self.model_path)

    def show_result(self):
        # 展示结果
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(self.best_fitness_history)), self.best_fitness_history)
        plt.xlabel('Epoch')
        plt.ylabel('Best Fitness')
        plt.title('LSHADE-Q Algorithm')
        plt.grid()
        plt.show()

# 定义适应度函数和约束条件
def fitness(x):
    return sum(xi ** 2 - 10 * cos(2 * pi * xi) + 10 for xi in x)

def constraints(x):
    g1 = sum(-xi * sin(2 * xi) for xi in x)
    g2 = sum(xi * sin(xi) for xi in x)
    return max(g1, 0) + max(g2, 0)

# 运行LSHADE-Q算法
lshade_q = LSHADE_Q(fitness, constraints, -100, 100, 100, 10, 1000)
lshade_q.run()
lshade_q.show_result()

