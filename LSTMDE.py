import numpy as np
import random
from math import cos, pi, sin
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os

class LSTM_DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM_DQN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out的形状为 (batch_size, seq_length, hidden_dim)
        return self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出

class LSHADE_Q:
    def __init__(self, fitness, constraints, lower, upper, pop_size, dim, epochs, seq_length, mode='train'):
        self.fitness = fitness
        self.constraints = constraints
        self.lowbound = lower
        self.upbound = upper
        self.pop_size = pop_size
        self.dim = dim
        self.Epochs = epochs
        self.NFE = 0
        self.max_NFE = 10000 * self.dim
        self.seq_length = seq_length  # 新增序列长度参数
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fit = np.random.rand(self.pop_size)
        self.conv = np.random.rand(self.pop_size)
        self.best = self.population[0]
        self.dqn = LSTM_DQN(input_dim=self.dim, hidden_dim=64, output_dim=2)  # 使用LSTM
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.model_path = 'lstm_dqn_model.pth'
        self.best_fitness_history = []
        self.mode = mode

    def initpop(self):
        self.population = self.lowbound + (self.upbound - self.lowbound) * np.random.rand(self.pop_size, self.dim)
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])
        self.conv = np.array([self.constraints(chrom) for chrom in self.population])
        self.NFE += self.pop_size
        self.best = self.population[np.argmin(self.fit)]

    def mut(self, i, F):
        idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 2, replace=False)
        F += np.random.uniform(-0.2, 0.2)
        F = max(0, min(F, 1))
        v = torch.tensor(self.population[idxs[0]], dtype=torch.float32, requires_grad=True) + F * (torch.tensor(self.population[idxs[1]], dtype=torch.float32, requires_grad=True) - torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True))
        return torch.clamp(v, self.lowbound, self.upbound)

    def cross(self, mut_chrom, i, CR):
        cross_chrom = torch.clone(torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True))
        j = np.random.randint(0, self.dim)
        for k in range(self.dim):
            if np.random.rand() < CR or k == j:
                cross_chrom[k] = mut_chrom[k]
        return cross_chrom

    def select(self, cross_chrom, i):
        temp = self.fitness(cross_chrom.detach().numpy())
        temp_v = self.constraints(cross_chrom.detach().numpy())
        self.NFE += 1
        if (self.conv[i] == 0 and temp_v == 0 and self.fit[i] >= temp) or (self.conv[i] > temp_v):
            self.population[i] = cross_chrom.detach().numpy()
            self.fit[i] = temp
            self.conv[i] = temp_v

    def run(self):
        if os.path.exists(self.model_path):
            try:
                self.dqn.load_state_dict(torch.load(self.model_path, weights_only=True))
                print("加载预训练模型成功...")
            except Exception as e:
                print(f"加载模型时发生错误: {e}")
                return

            if self.mode == '1':  # 预测
                self.fit = np.array([self.fitness(chrom) for chrom in self.population])
                self.conv = np.array([self.constraints(chrom) for chrom in self.population])

                for epoch in range(self.Epochs):
                    for i in range(self.pop_size):
                        if epoch < self.seq_length:
                            # 如果历史信息不足，填充零
                            state_sequence = np.zeros((self.seq_length, self.dim))
                            state_sequence[-(epoch + 1):] = self.population[i]  # 填充当前状态
                        else:
                            # 使用过去的状态
                            state_sequence = self.population[i - self.seq_length + 1:i + 1]

                        # 确保状态序列的长度为seq_length
                        if len(state_sequence) < self.seq_length:
                            state_sequence = np.pad(state_sequence,
                                                    ((self.seq_length - len(state_sequence), 0), (0, 0)), 'constant')

                        # 确保 state_sequence 的形状为 (seq_length, dim)
                        if state_sequence.shape[0] == 0:
                            print(f"Warning: State sequence for epoch {epoch}, individual {i} is empty.")
                            state_sequence = np.zeros((self.seq_length, self.dim))  # 用零填充

                        state = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0)  # (1, seq_length, dim)

                        # 确保输入到LSTM的形状是正确的
                        if state.shape[1] != self.seq_length or state.shape[2] != self.dim:
                            print(f"Error: Invalid state shape {state.shape} for epoch {epoch}, individual {i}")
                            continue

                        action = self.dqn(state)

                        F = action[0, 0].item() * (self.upbound - self.lowbound) + self.lowbound
                        CR = action[0, 1].item()
                        CR = max(0, min(CR, 1))  # 限制CR的范围

                        # 变异
                        mut_chrom = self.mut(i, F)
                        # 交叉
                        cross_chrom = self.cross(mut_chrom, i, CR)
                        # 计算适应度和约束
                        temp_fit = self.fitness(cross_chrom.detach().numpy())
                        temp_conv = self.constraints(cross_chrom.detach().numpy())
                        self.NFE += 1

                        # 选择
                        self.select(cross_chrom, i)

                    # 更新最优个体
                    self.best = self.population[np.argmin(self.fit)]
                    print(f'Epoch: {epoch}, 最佳适应度: {np.min(self.fit)}, 最优个体: {self.best}')

                return  # 结束运行，不进行训练

        print("预训练模型不存在，开始训练...")
        self.initpop()

        # 进行迭代
        for epoch in range(self.Epochs):
            for i in range(self.pop_size):
                if epoch < self.seq_length:
                    state_sequence = np.zeros((self.seq_length, self.dim))
                    state_sequence[-(epoch + 1):] = self.population[i]  # 填充当前状态
                else:
                    start_index = max(0, i - self.seq_length + 1)
                    state_sequence = self.population[start_index:i + 1]

                    # 确保状态序列的长度为seq_length
                    if len(state_sequence) < self.seq_length:
                        state_sequence = np.pad(state_sequence, ((self.seq_length - len(state_sequence), 0), (0, 0)),
                                                'constant')

                # 确保 state_sequence 的形状为 (seq_length, dim)
                if state_sequence.shape[0] == 0:
                    print(f"Warning: State sequence for epoch {epoch}, individual {i} is empty.")
                    state_sequence = np.zeros((self.seq_length, self.dim))  # 用零填充


                state = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0)  # (1, seq_length, dim)

                # 确保输入到LSTM的形状是正确的
                if state.shape[1] != self.seq_length or state.shape[2] != self.dim:
                    print(f"Error: Invalid state shape {state.shape} for epoch {epoch}, individual {i}")
                    continue

                action = self.dqn(state)

                F = action[0, 0].item() * (self.upbound - self.lowbound) + self.lowbound
                CR = action[0, 1].item()
                CR = max(0, min(CR, 1))  # 限制CR的范围

                # 变异
                mut_chrom = self.mut(i, F)
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
        plt.title('LSHADE-Q Algorithm with LSTM')
        plt.grid()
        plt.show()

# 定义适应度函数和约束条件
def fitness(x):
    return sum(xi ** 2 - 10 * cos(2 * pi * xi) + 10 for xi in x)

def constraints(x):
    g1 = sum(-xi * sin(2 * xi) for xi in x)
    g2 = sum(xi * sin(xi) for xi in x)
    return max(g1, 0) + max(g2, 0)

# 在创建LSHADE_Q实例时，设置seq_length
seq_length = 10  # 例如，使用过去5个状态
lshade_q = LSHADE_Q(fitness, constraints, -100, 100, 100, 10, 1000, seq_length, mode='1')
lshade_q.run()
lshade_q.show_result()

