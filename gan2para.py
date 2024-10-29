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
from sklearn.preprocessing import StandardScaler

# 设置matplotlib的字体为SimHei,以正确显示中文
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class DuelingDQN(nn.Module):
    """
    Dueling DQN网络结构
    将Q值分解为状态价值函数V(s)和优势函数A(s,a)
    """
    def __init__(self, input_dim, output_dim=1):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q = value + (advantage - advantage.mean())
        return q  # 输出形状为 (batch_size, output_dim)

class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    根据TD误差的绝对值来确定样本的优先级
    """
    def __init__(self, capacity=10000, alpha=0.6, device=torch.device('cpu')):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # 决定优先级的影响程度
        self.epsilon = 1e-6  # 防止优先级为0
        self.device = device

    def push(self, state, action, reward, next_state, done):
        # 新样本的优先级设为当前最大优先级
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # 根据优先级进行采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        batch = list(zip(*samples))
        states, actions, rewards, next_states, dones = map(np.array, batch)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return len(self.buffer)

class LSHADE_DQN:
    """
    LSHADE算法与DQN结合的实现
    用于解决多目标优化问题
    """
    def __init__(self, fitness, constraints, lower, upper, pop_size, dim, epochs,
                 gamma, learning_rate, epsilon_start, epsilon_end, epsilon_decay,
                 alpha, beta_start, beta_frames, buffer_size, batch_size, target_update):
        # 初始化各种参数
        self.fitness = fitness
        self.constraints = constraints
        self.lower = lower
        self.upper = upper
        self.pop_size = pop_size
        self.dim = dim
        self.epochs = epochs
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化种群
        self.population = np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))
        self.fit = np.array([self.fitness(ind) for ind in self.population])
        self.conv = np.array([self.constraints(ind) for ind in self.population])

        # 初始化标准化器
        self.scaler = StandardScaler()
        self.scaler.fit(self.population)

        # 初始化Dueling DQN模型和优化器
        self.dqn_F = DuelingDQN(input_dim=self.dim, output_dim=1).to(self.device)
        self.dqn_CR = DuelingDQN(input_dim=self.dim, output_dim=1).to(self.device)
        self.target_dqn_F = DuelingDQN(input_dim=self.dim, output_dim=1).to(self.device)
        self.target_dqn_CR = DuelingDQN(input_dim=self.dim, output_dim=1).to(self.device)
        self.target_dqn_F.load_state_dict(self.dqn_F.state_dict())
        self.target_dqn_CR.load_state_dict(self.dqn_CR.state_dict())
        self.target_dqn_F.eval()
        self.target_dqn_CR.eval()

        self.optimizer_F = optim.Adam(self.dqn_F.parameters(), lr=self.learning_rate)
        self.optimizer_CR = optim.Adam(self.dqn_CR.parameters(), lr=self.learning_rate)

        self.criterion = nn.MSELoss(reduction='none')  # 计算每个样本的损失

        # 初始化经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(capacity=self.buffer_size, alpha=self.alpha, device=self.device)

        # 加载真实Pareto前沿（如果存在）
        self.true_pareto_front = self.load_true_pareto_front()

        # 初始化Pareto前沿
        self.pareto_front = self.calculate_pareto_front()
        self.frame_idx = 0  # 用于beta的动态调整

        # 初始化epsilon
        self.epsilon = self.epsilon_start

    def load_true_pareto_front(self):
        """加载真实Pareto前沿"""
        if os.path.exists('true_pareto_front.mat'):
            data = loadmat('true_pareto_front.mat')
            return data['PF']
        else:
            print("真实Pareto前沿文件不存在，初始化为空集合。")
            return np.empty((0, 2))

    def calculate_pareto_front(self):
        """计算当前种群的Pareto前沿"""
        non_dominated = []
        for i, fit_i in enumerate(self.fit):
            dominated = False
            for j, fit_j in enumerate(self.fit):
                if i != j:
                    if np.all(fit_j <= fit_i) and np.any(fit_j < fit_i):
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(fit_i)
        return np.array(non_dominated)

    def calculate_hypervolume(self, pareto_front, reference_point=np.array([1.1, 1.1])):
        """计算Pareto前沿的超体积"""
        if pareto_front.size == 0:
            return 0.0
        # 由于f1和f2都是最小化的，这里假设目标是要尽量靠近(0,0)
        sorted_pf = pareto_front[pareto_front[:, 0].argsort()]
        hypervolume = 0.0
        previous_f1 = 0.0
        for f1, f2 in sorted_pf:
            hypervolume += (f2) * (f1 - previous_f1)
            previous_f1 = f1
        hypervolume += (reference_point[0] - previous_f1) * reference_point[1]
        return hypervolume

    def calculate_diversity_reward(self, population):
        """计算多样性奖励，基于超体积和帕累托前沿的多样性"""
        pareto = self.calculate_pareto_front()
        hypervolume = self.calculate_hypervolume(pareto)
        return hypervolume

    def is_dominated(self, new_fit, current_fit):
        """判断new_fit是否支配current_fit"""
        return np.all(new_fit <= current_fit) and np.any(new_fit < current_fit)

    def choose_action(self, state, epsilon):
        """
        选择动作F和CR。使用两个独立的DQN模型分别选择F和CR。
        """
        actions_F = []
        actions_CR = []
        for s in state:
            if random.random() < epsilon:
                # 探索：随机选择F和CR
                F = np.random.uniform(0, 1)
                CR = np.random.uniform(0, 1)
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                    F = self.dqn_F(s_tensor).cpu().numpy()[0][0]
                    CR = self.dqn_CR(s_tensor).cpu().numpy()[0][0]
                    # 确保F和CR在[0,1]范围内
                    F = np.clip(F, self.lower, self.upper)
                    CR = np.clip(CR, self.lower, self.upper)
            actions_F.append(F)
            actions_CR.append(CR)
        actions_F = np.array(actions_F).reshape(-1, 1)
        actions_CR = np.array(actions_CR).reshape(-1, 1)
        actions = np.hstack([actions_F, actions_CR])  # 形状 [pop_size, 2]
        return torch.tensor(actions, dtype=torch.float32).to(self.device)

    def mutate(self, idx, F):
        """变异操作，使用Differential Evolution策略"""
        idxs = list(range(0, self.pop_size))
        idxs.remove(idx)
        a, b, c = random.sample(idxs, 3)
        mutant = self.population[a] + F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower, self.upper)
        return mutant

    def crossover(self, parent, mutant, CR):
        """交叉操作，结合parent和mutant生成trial个体"""
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, parent)
        return trial

    def select(self, idx, trial, new_fit, new_conv):
        """选择操作，根据适应度和约束决定是否接受试验解"""
        if new_conv <= self.conv[idx]:
            if self.is_dominated(new_fit, self.fit[idx]):
                self.population[idx] = trial
                self.fit[idx] = new_fit
                self.conv[idx] = new_conv

    def update_pareto_front(self):
        """更新Pareto前沿"""
        self.pareto_front = self.calculate_pareto_front()

    def learn(self):
        """从经验回放缓冲区采样并执行学习步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None  # 不足批量大小，不进行学习

        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        try:
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size,
                                                                                                       beta=beta)
        except ValueError as e:
            print(f"经验回放缓冲区采样错误: {e}")
            return None, None

        # 标准化
        states = self.scaler.transform(states)
        next_states = self.scaler.transform(next_states)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)  # 采取的动作 [F, CR]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)  # 奖励，形状 [batch_size,1]
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)  # 下一个状态
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)  # 终止标志，形状 [batch_size,1]
        weights = weights.to(self.device)

        # 分离F和CR动作
        actions_F = actions[:, 0].unsqueeze(1)  # [batch_size,1]
        actions_CR = actions[:, 1].unsqueeze(1)  # [batch_size,1]

        #### 计算F的损失 ####
        current_Q_F = self.dqn_F(states)  # 形状 [batch_size,1]
        with torch.no_grad():
            target_Q_F = self.target_dqn_F(next_states)  # 形状 [batch_size,1]
            target_F = rewards + self.gamma * target_Q_F * (1 - dones)  # 形状 [batch_size,1]
        loss_F = self.criterion(current_Q_F, target_F).squeeze()  # [batch_size]

        #### 计算CR的损失 ####
        current_Q_CR = self.dqn_CR(states)  # 形状 [batch_size,1]
        with torch.no_grad():
            target_Q_CR = self.target_dqn_CR(next_states)  # 形状 [batch_size,1]
            target_CR = rewards + self.gamma * target_Q_CR * (1 - dones)  # 形状 [batch_size,1]
        loss_CR = self.criterion(current_Q_CR, target_CR).squeeze()  # [batch_size]

        # 合并F和CR的损失
        per_sample_loss = loss_F + loss_CR  # [batch_size]
        weighted_loss = per_sample_loss * weights.squeeze()  # [batch_size]

        loss = weighted_loss.mean()

        # 反向传播和优化
        self.optimizer_F.zero_grad()
        self.optimizer_CR.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_F.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.dqn_CR.parameters(), max_norm=1.0)
        self.optimizer_F.step()
        self.optimizer_CR.step()

        # 更新优先级
        priorities = per_sample_loss.detach().cpu().numpy() + 1e-6  # [batch_size]
        self.replay_buffer.update_priorities(indices, priorities)

        # 更新目标网络
        if self.frame_idx % self.target_update == 0:
            self.target_dqn_F.load_state_dict(self.dqn_F.state_dict())
            self.target_dqn_CR.load_state_dict(self.dqn_CR.state_dict())

        self.frame_idx += 1

        return loss_F.mean().item(), loss_CR.mean().item()  # 返回当前损失值

    def run(self):
        no_improve_epochs = 0
        old_diversity_reward = self.calculate_diversity_reward(self.population)  # 用于多样性分数的计算和保存

        for epoch in range(1, self.epochs + 1):
            epoch_reward = 0
            epoch_no_improve_pop = 0

            # 标准化
            scaled_population = self.scaler.transform(self.population)

            # 选择动作，获取 F 和 CR
            actions = self.choose_action(scaled_population, self.epsilon)  # 形状 [pop_size, 2]
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

            # 更新Pareto前沿
            self.update_pareto_front()

            # 计算多样性奖励的变化
            new_diversity_reward = self.calculate_diversity_reward(self.population)
            epoch_reward = new_diversity_reward - old_diversity_reward
            old_diversity_reward = new_diversity_reward  # 更新旧的多样性奖励

            print(f'第 {epoch} 轮, 超体积增益: {epoch_reward:.4f}')

            # 检查是否有改进
            if epoch_reward > 1e-6:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # 衰减epsilon
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           np.exp(-1. * epoch / self.epsilon_decay)

            # 执行学习步骤
            loss = self.learn()
            if loss is not None:
                loss_F, loss_CR = loss
                print(f"训练损失F: {loss_F:.4f}, 训练损失CR: {loss_CR:.4f}")

            if no_improve_epochs > 20:
                print("超过允许的无改进轮数，提前停止训练。")
                break

        # 训练完成后保存模型
        torch.save(self.dqn_F.state_dict(), 'dqn_F_model.pth')
        torch.save(self.dqn_CR.state_dict(), 'dqn_CR_model.pth')
        print("模型已保存到:", 'dqn_F_model.pth', "和", 'dqn_CR_model.pth')

    def show_result(self):
        """展示Pareto前沿和种群的适应度"""
        plt.figure(figsize=(10, 6))

        # 绘制真实Pareto前沿
        if self.true_pareto_front.size > 0:
            plt.scatter(self.true_pareto_front[:, 0], self.true_pareto_front[:, 1],
                        color='green', label='真实 Pareto 前沿', marker='x')

        # 绘制Pareto前沿
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
    # 初始化LSHADE_DQN实例并运行
    lshade_dqn = LSHADE_DQN(
        fitness=fitness,  # 适应度函数
        constraints=constraints,  # 约束条件
        lower=0.0,  # ZDT1定义域下限，通常为0
        upper=1.0,  # ZDT1定义域上限，通常为1
        pop_size=100,  # 种群规模，适应高维优化问题
        dim=10,  # 设置维度为10
        epochs=100,  # 训练轮数，增加以提升性能
        gamma=0.99,  # 折扣因子，用于未来奖励的权重
        learning_rate=0.0005,  # 学习率，较小以稳定训练
        epsilon_start=1.0,  # 初始探索率，高度探索
        epsilon_end=0.05,  # 最小探索率，确保一定程度的探索
        epsilon_decay=300,  # 探索率衰减速度
        alpha=0.6,  # 优先级回放中的α参数
        beta_start=0.4,  # 重要性采样中的β初始值
        beta_frames=1000,  # β衰减的帧数
        buffer_size=10000,  # 经验回放缓冲区大小
        batch_size=64,  # 批量大小，增加以利用更多数据
        target_update=1000  # 目标网络更新频率
    )

    lshade_dqn.run()
    lshade_dqn.show_result()