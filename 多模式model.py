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
    def __init__(self, fitness, constraints, lower, upper, pop_size, dim, epochs, mode='train'):
        self.fitness = fitness
        self.constraints = constraints
        self.lowbound = lower
        self.upbound = upper
        self.pop_size = pop_size
        self.dim = dim
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fit = np.random.rand(self.pop_size)
        self.conv = np.random.rand(self.pop_size)
        self.best = self.population[0]
        self.Epochs = epochs
        self.NFE = 0
        self.max_NFE = 10000 * self.dim
        self.dqn = DQN(input_dim=self.dim, output_dim=2)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.model_path = 'dqn_model.pth'
        self.best_fitness_history = []
        self.mode = mode  # 新增模式标志

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
        F += np.random.uniform(-0.2, 0.2)  # 增加变异范围
        F = max(0, min(F, 1))  # 限制F的范围
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
            if np.random.rand() < CR or k == j:
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
            try:
                self.dqn.load_state_dict(torch.load(self.model_path, weights_only=True))
                print("加载预训练模型成功...")
            except Exception as e:
                print(f"加载模型时发生错误: {e}")
                return

            if self.mode == 'predict':
                # 初始化适应度
                self.fit = np.array([self.fitness(chrom) for chrom in self.population])
                self.conv = np.array([self.constraints(chrom) for chrom in self.population])

                for epoch in range(self.Epochs):
                    for i in range(self.pop_size):
                        state = torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True)
                        action = self.dqn(state)

                        if action is None or not isinstance(action, torch.Tensor) or action.shape[0] != 2:
                            print("DQN模型输出无效，跳过此个体。")
                            continue

                        F = action[0].item() * (self.upbound - self.lowbound) + self.lowbound
                        CR = action[1].item()
                        CR = max(0, min(CR, 1))  # 限制CR的范围

                        # 变异
                        mut_chrom = self.mut(i, F)  # 传递F
                        # 交叉
                        cross_chrom = self.cross(mut_chrom, i, CR)
                        # 计算适应度和约束
                        temp_fit = self.fitness(cross_chrom.detach().numpy())
                        temp_conv = self.constraints(cross_chrom.detach().numpy())
                        self.NFE += 1  # 更新函数评价次数

                        # 选择
                        if (self.conv[i] == 0 and temp_conv == 0 and self.fit[i] >= temp_fit) or (
                                self.conv[i] > temp_conv):
                            self.population[i] = cross_chrom.detach().numpy()
                            self.fit[i] = temp_fit
                            self.conv[i] = temp_conv

                    # 更新最优个体
                    self.best = self.population[np.argmin(self.fit)]
                    print(f'Epoch: {epoch}, 最佳适应度: {np.min(self.fit)}, 最优个体: {self.best}')

                return  # 结束运行，不进行训练

        print("预训练模型不存在，开始训练...")
        # 如果没有预训练模型，则初始化种群并进行训练
        self.initpop()

        # 进行迭代
        for epoch in range(self.Epochs):
            for i in range(self.pop_size):
                state = torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True)
                action = self.dqn(state)

                if action is None or not isinstance(action, torch.Tensor):
                    print("DQN模型输出无效，跳过此个体。")
                    continue

                F = action[0].item() * (self.upbound - self.lowbound) + self.lowbound
                CR = action[1].item()
                CR = max(0, min(CR, 1))  # 限制CR的范围

                # 变异
                mut_chrom = self.mut(i, F)  # 传递F
                # 交叉
                cross_chrom = self.cross(mut_chrom, i, CR)
                # 选择
                self.select(cross_chrom, i)

            # 更新最优个体
            self.best = self.population[np.argmin(self.fit)]
            self.best_fitness_history.append(np.min(self.fit))
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

mode = '1'  # 或者 'predict'，根据需要选择
lshade_q = LSHADE_Q(fitness, constraints, -100, 100, 100, 10, 1000, mode)
lshade_q.run()
lshade_q.show_result()

