import numpy as np
import random
from math import cos, pi, sin
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

class LSHADE_Q:
    def __init__(self, fitness, constraints, lower, upper, pop_size, dim, epochs):
        self.fitness = fitness
        self.constraints = constraints
        self.lowbound = lower
        self.upbound = upper
        self.pop_size = pop_size
        self.dim = dim
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fit = np.random.rand(self.pop_size, 2)  # 适应度现在是二维的
        self.conv = np.random.rand(self.pop_size)
        self.best = self.population[0]
        self.Epochs = epochs
        self.NFE = 0
        self.max_NFE = 10000 * self.dim
        self.dqn = DQN(input_dim=self.dim, output_dim=2)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.model_path = 'dqn_model.pth'
        self.best_fitness_history = []

    def initpop(self):
        self.population = self.lowbound + (self.upbound - self.lowbound) * np.random.rand(self.pop_size, self.dim)
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])
        self.conv = np.array([self.constraints(chrom) for chrom in self.population])
        self.NFE += self.pop_size
        self.best = self.population[np.argmin(self.fit[:, 0])]  # 选择第一个目标的最优个体

    def mut(self, i, F):
        idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 2, replace=False)
        F = torch.clamp(F + np.random.uniform(-0.2, 0.2), 0, 1)
        v = torch.tensor(self.population[idxs[0]], dtype=torch.float32, requires_grad=True) + F * (torch.tensor(self.population[idxs[1]], dtype=torch.float32, requires_grad=True) - torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True))
        return torch.clamp(v, self.lowbound, self.upbound)

    def cross(self, mut_chrom, i, CR):
        cross_chrom = torch.clone(torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True))
        j = np.random.randint(0, self.dim)
        for k in range(self.dim):
            if np.random.rand() < CR.detach().numpy() or k == j:
                cross_chrom[k] = mut_chrom[k]
        return cross_chrom

    def pareto_dominance(self, a, b):
        """检查个体a是否支配个体b"""
        return np.all(a <= b) and np.any(a < b)

    def select(self, cross_chrom, i):
        temp = self.fitness(cross_chrom.detach().numpy())
        temp_v = self.constraints(cross_chrom.detach().numpy())
        self.NFE += 1
        if (self.conv[i] == 0 and temp_v == 0 and np.all(self.fit[i] >= temp)) or (self.conv[i] > temp_v):
            self.population[i] = cross_chrom.detach().numpy()
            self.fit[i] = temp
            self.conv[i] = temp_v

    def run(self):
        if os.path.exists(self.model_path):
            self.dqn.load_state_dict(torch.load(self.model_path))
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
                new_fit = self.fitness(cross_chrom.detach().numpy())
                old_fit = self.fitness(self.population[i])
                reward = torch.tensor(np.sum(new_fit) - np.sum(old_fit), dtype=torch.float32, requires_grad=True)

                # 计算损失
                loss = -reward
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 更新最优个体
            self.best = self.population[np.argmin(self.fit[:, 0])]  # 选择第一个目标的最优个体
            # 存储最佳适应度
            self.best_fitness_history.append(np.min(self.fit[:, 0]))  # 记录第一个目标的最佳适应度

            # 获取当前最佳个体的适应度
            best_fitness = self.fit[np.argmin(self.fit[:, 0])]

            # 打印信息
            print(
                f'Epoch: {epoch}, Best Fitness: {best_fitness[0]}, Second Objective: {best_fitness[1]}, Best Individual: {self.best}')

        # 训练完成后保存模型
        torch.save(self.dqn.state_dict(), self.model_path)
        print("模型已保存到:", self.model_path)

    def show_result(self):
        # 展示结果
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(self.best_fitness_history)), self.best_fitness_history)
        plt.xlabel('Epoch')
        plt.ylabel('Best Fitness (Objective 1)')
        plt.title('LSHADE-Q Algorithm (Multi-Objective)')
        plt.grid()
        plt.show()

# 定义多目标适应度函数
def fitness(x):
    # 目标1: 最小化
    obj1 = sum(xi ** 2 - 10 * cos(2 * pi * xi) + 10 for xi in x)
    # 目标2: 最小化
    obj2 = sum((xi - 5) ** 2 for xi in x)  # 例如，最小化到5的距离
    return np.array([obj1, obj2])  # 返回多个目标

def constraints(x):
    g1 = sum(-xi * sin(2 * xi) for xi in x)
    g2 = sum(xi * sin(xi) for xi in x)
    return max(g1, 0) + max(g2, 0)

lshade_q = LSHADE_Q(fitness, constraints, -100, 100, 100, 10, 1000)
lshade_q.run()
lshade_q.show_result()

