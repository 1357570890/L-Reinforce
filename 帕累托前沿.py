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
        self.target_dqn = DQN(input_dim=self.dim, output_dim=2)  # 目标网络
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.model_path = 'dqn_model.pth'
        self.best_fitness_history = []
        self.update_target_steps = 4  # 更新目标网络的步数

        # 初始化目标网络
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()  # 目标网络设置为评估模式

    def initpop(self):
        self.population = self.lowbound + (self.upbound - self.lowbound) * np.random.rand(self.pop_size, self.dim)
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])
        self.conv = np.array([self.constraints(chrom) for chrom in self.population])
        self.NFE += self.pop_size
        self.update_pareto_front()

    def update_pareto_front(self):
        # 更新帕累托前沿
        pareto_front = []
        for i in range(self.pop_size):
            is_dominated = False
            for j in range(self.pop_size):
                if i != j and np.all(self.fit[j] <= self.fit[i]) and np.any(self.fit[j] < self.fit[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(self.fit[i])  # 存储适应度而不是个体
        self.pareto_front = np.array(pareto_front)
        #print(f"当前帕累托前沿坐标: {self.pareto_front}")  # 打印帕累托前沿坐标

    def mut(self, i, F):
        idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 2, replace=False)
        F = torch.clamp(F + np.random.uniform(-0.2, 0.2), 0, 1)
        v = torch.tensor(self.population[idxs[0]], dtype=torch.float32, requires_grad=True) + F * (torch.tensor(self.population[idxs[1]], dtype=torch.float32, requires_grad=True) - torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True))
        return torch.clamp(v, self.lowbound, self.upbound)

    def cross(self, mut_chrom, i, CR):
        cross_chrom = torch.clone(torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True))
        j = np.random.randint(0, self.dim)  # 随机选择一个交叉点
        for k in range(self.dim):
            if np.random.rand() < CR.detach().numpy() or k == j:
                cross_chrom[k] = mut_chrom[k]  # 用变异个体的基因替换当前个体的基因
        return cross_chrom

    def select(self, cross_chrom, i):
        temp = self.fitness(cross_chrom.detach().numpy())
        temp_v = self.constraints(cross_chrom.detach().numpy())
        self.NFE += 1
        if (self.conv[i] == 0 and temp_v == 0 and np.all(self.fit[i] >= temp)) or (self.conv[i] > temp_v):
            self.population[i] = cross_chrom.detach().numpy()
            self.fit[i] = temp
            self.conv[i] = temp_v
            self.update_pareto_front()  # 更新帕累托前沿

    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def run(self):
        try:
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
                    try:
                        # 获取当前状态
                        state = torch.tensor(self.population[i], dtype=torch.float32, requires_grad=True)
                        action = self.dqn(state)
                        if action is None:
                            print("DQN模型输出为 None，跳过此个体。")
                            continue

                        F = torch.sigmoid(action[0])  # 使用 sigmoid 确保 F 在 [0, 1] 范围内
                        CR = torch.clamp(action[1], 0, 1)  # 确保 CR 在 [0, 1] 范围内

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

                        # 计算目标值
                        with torch.no_grad():
                            target_action = self.target_dqn(cross_chrom)
                            target_value = target_action.max().item()  # 选择最大值作为目标

                        # 计算损失
                        loss = -reward + target_value  # 这里的损失可以根据需要调整
                        # 反向传播
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    except Exception as e:
                        print(f"个体 {i} 处理时发生错误: {e}")

                # 更新最优个体
                self.update_pareto_front()  # 更新帕累托前沿
                # 存储最佳适应度
                self.best_fitness_history.append(np.min(self.fit[:, 0]))  # 记录第一个目标的最佳适应度

                # 打印信息
                print(f'Epoch: {epoch}, Current Pareto Front Size: {len(self.pareto_front)}')

                # 每隔一定步数更新目标网络
                if epoch % self.update_target_steps == 0:
                    self.update_target_network()

            # 训练完成后保存模型
            torch.save(self.dqn.state_dict(), self.model_path)
            print("模型已保存到:", self.model_path)

        except Exception as e:
            print(f"运行过程中发生错误: {e}")

    def show_result(self):
        # 展示结果
        plt.figure(figsize=(10, 6))

        # 绘制帕累托前沿
        if self.pareto_front.size > 0:
            plt.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1], color='blue', label='Pareto Front', alpha=0.5)

        # 绘制所有个体的适应度
        plt.scatter(self.fit[:, 0], self.fit[:, 1], color='red', label='All Individuals', alpha=0.3)

        # 标记最佳个体
        best_index = np.argmin(self.fit[:, 0])  # 找到第一个目标的最佳个体
        best_fitness = self.fit[best_index]
        plt.scatter(best_fitness[0], best_fitness[1], color='green', s=100, label='Best Individual', edgecolor='black')

        # 设置图形的标签和标题
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front and Individuals')
        plt.legend()
        plt.grid()
        plt.show()

        # 打印帕累托前沿的具体坐标
        print("帕累托前沿点的具体坐标:")
        for point in self.pareto_front:
            print(point)


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

if __name__ == "__main__":
    lshade_q = LSHADE_Q(fitness, constraints, -100, 100, 100, 10, 1000)
    lshade_q.run()
    lshade_q.show_result()

