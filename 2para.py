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


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 保持输出在 [0,1] 范围内
        return x  # 输出形状为 (batch_size, output_dim)


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

        # 初始化两个DQN，一个用于F，一个用于CR
        self.dqn_F = DQN(input_dim=self.dim, output_dim=1).to(self.get_device())
        self.dqn_CR = DQN(input_dim=self.dim, output_dim=1).to(self.get_device())
        self.optimizer_F = optim.Adam(self.dqn_F.parameters(), lr=learning_rate)
        self.optimizer_CR = optim.Adam(self.dqn_CR.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # 均方误差损失函数
        self.model_path_F = 'dqn_F_model.pth'
        self.model_path_CR = 'dqn_CR_model.pth'

        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # 初始化Pareto前沿
        self.update_pareto_front()

        # 加载真实帕累托前沿
        self.load_true_pareto_front("true_pareto_front.mat")

        # 加载预训练模型（如果存在）
        self.load_models()

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self):
        # 加载F模型
        if os.path.exists(self.model_path_F):
            try:
                self.dqn_F.load_state_dict(torch.load(self.model_path_F, map_location=self.get_device()))
                print("加载预训练模型F成功，继续训练...")
            except Exception as e:
                print(f"加载预训练模型F失败: {e}")
                print("将使用新初始化的模型F开始训练...")
        else:
            print("预训练模型F不存在，开始训练...")

        # 加载CR模型
        if os.path.exists(self.model_path_CR):
            try:
                self.dqn_CR.load_state_dict(torch.load(self.model_path_CR, map_location=self.get_device()))
                print("加载预训练模型CR成功，继续训练...")
            except Exception as e:
                print(f"加载预训练模型CR失败: {e}")
                print("将使用新初始化的模型CR开始训练...")
        else:
            print("预训练模型CR不存在，开始训练...")
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
        # 简单选择，若新个体更好则替换
        if new_conv <= self.conv[i]:
            if np.all(new_fit <= self.fit[i]):
                self.population[i] = trial
                self.fit[i] = new_fit
                self.conv[i] = new_conv

    def calculate_diversity_reward(self, population):
        # 计算种群多样性的简单方法：种群解的平均距离
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distances.append(np.linalg.norm(population[i] - population[j]))
        if len(distances) == 0:
            return 0.0
        return np.mean(distances)

    def choose_action(self, state):
        """
        选择动作F和CR。使用两个独立的DQN模型分别选择F和CR。
        """
        actions_F = []
        actions_CR = []
        for s in state:
            if random.random() < self.epsilon:
                # 探索：随机选择F和CR
                F = np.random.uniform(0, 1)
                CR = np.random.uniform(0, 1)
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.get_device())
                    F = self.dqn_F(s_tensor).cpu().numpy()[0][0]
                    CR = self.dqn_CR(s_tensor).cpu().numpy()[0][0]
                    # 确保F和CR在[0,1]范围内
                    F = np.clip(F, self.lowbound, self.upbound)
                    CR = np.clip(CR, self.lowbound, self.upbound)
            actions_F.append(F)
            actions_CR.append(CR)
        actions_F = np.array(actions_F).reshape(-1, 1)
        actions_CR = np.array(actions_CR).reshape(-1, 1)
        actions = np.hstack([actions_F, actions_CR])  # 形状 [pop_size, 2]
        return torch.tensor(actions, dtype=torch.float32).to(self.get_device())

    def learn(self):
        """从经验回放缓冲区采样并执行学习步骤"""

        if len(self.replay_buffer) < self.batch_size:
            return None, None  # 不足批量大小，不进行学习

        # 从经验回放缓冲区中随机采样一批数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 将采样的数据转换为PyTorch张量，并移动到指定设备（如GPU）
        states = torch.tensor(states, dtype=torch.float32).to(self.get_device())  # 当前状态
        actions = torch.tensor(actions, dtype=torch.float32).to(self.get_device())  # 采取的动作 [F, CR]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.get_device())  # 奖励，形状 [batch_size,1]
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.get_device())  # 下一个状态
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.get_device())  # 终止标志，形状 [batch_size,1]

        # 分离F和CR动作
        actions_F = actions[:, 0].unsqueeze(1)  # [batch_size,1]
        actions_CR = actions[:, 1].unsqueeze(1)  # [batch_size,1]

        #### 学习F ####
        current_Q_F = self.dqn_F(states)  # 形状 [batch_size,1]

        with torch.no_grad():
            target_Q_F = self.dqn_F(next_states)  # 形状 [batch_size,1]
            target_F = rewards + self.gamma * target_Q_F * (1 - dones)  # 形状 [batch_size,1]

        # 计算损失
        loss_F = self.criterion(current_Q_F, target_F)

        # 优化前记录参数的范数
        param_norm_before_F = 0.0
        for param in self.dqn_F.parameters():
            param_norm_before_F += param.data.norm(2).item()

        # 优化模型
        self.optimizer_F.zero_grad()
        loss_F.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_F.parameters(), max_norm=1.0)
        self.optimizer_F.step()

        # 优化后记录参数的范数
        param_norm_after_F = 0.0
        for param in self.dqn_F.parameters():
            param_norm_after_F += param.data.norm(2).item()

        print(f"DQN_F 参数范数变化前: {param_norm_before_F:.4f}, 变化后: {param_norm_after_F:.4f}, 差异: {param_norm_after_F - param_norm_before_F:.6f}")

        #### 学习CR ####
        current_Q_CR = self.dqn_CR(states)  # 形状 [batch_size,1]

        with torch.no_grad():
            target_Q_CR = self.dqn_CR(next_states)  # 形状 [batch_size,1]
            target_CR = rewards + self.gamma * target_Q_CR * (1 - dones)  # 形状 [batch_size,1]

        # 计算损失
        loss_CR = self.criterion(current_Q_CR, target_CR)

        # 优化前记录参数的范数
        param_norm_before_CR = 0.0
        for param in self.dqn_CR.parameters():
            param_norm_before_CR += param.data.norm(2).item()

        # 优化模型
        self.optimizer_CR.zero_grad()
        loss_CR.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_CR.parameters(), max_norm=1.0)
        self.optimizer_CR.step()

        # 优化后记录参数的范数
        param_norm_after_CR = 0.0
        for param in self.dqn_CR.parameters():
            param_norm_after_CR += param.data.norm(2).item()

        print(f"DQN_CR 参数范数变化前: {param_norm_before_CR:.4f}, 变化后: {param_norm_after_CR:.4f}, 差异: {param_norm_after_CR - param_norm_before_CR:.6f}")

        return loss_F.item(), loss_CR.item()  # 返回当前损失值

    def run(self):
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

            actions = self.choose_action(state)  # 形状 [pop_size, 2]
            actions_np = actions.cpu().numpy()

            for i in range(self.pop_size):
                F, CR = actions_np[i]
                mutant = self.mutate(i, F)
                trial = self.crossover(self.population[i], mutant, CR)
                new_fit = self.fitness(trial)
                new_conv = self.constraints(trial)
                self.select(i, trial, new_fit, new_conv)

                # 添加经验到回放缓冲区
                done = False  # ZDT1 无终止条件
                reward = self.calculate_diversity_reward(self.population)  # 使用多样性奖励
                self.replay_buffer.push(self.population[i], [F, CR], reward, self.population[i], done)

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
            loss = self.learn()
            if loss is not None:
                loss_F, loss_CR = loss
                print(f"训练损失F: {loss_F:.4f}, 训练损失CR: {loss_CR:.4f}")

            if no_improve_epochs > 10:
                print("超过允许的无改进轮数，提前停止训练。")
                break

        # 训练完成后保存模型
        torch.save(self.dqn_F.state_dict(), self.model_path_F)
        torch.save(self.dqn_CR.state_dict(), self.model_path_CR)
        print("模型已保存到:", self.model_path_F, "和", self.model_path_CR)

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
                        color='blue', label='获得的 Pareto 前沿', alpha=0.6)
        # 绘制所有个体的适应度
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


def fitness(x):
    """计算ZDT1的多目标适应度值"""
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - sqrt(f1 / g))
    return np.array([f1, f2])


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
        pop_size=100,  # 种群规模，减少以适应低配置电脑
        dim=10,  # 设置维度为10
        epochs=100,  # 训练轮数，减少以加快训练速度
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