import numpy as np
import random
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D
import os
from collections import deque
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPONetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim * 2)  # 输出均值和标准差
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        actor_output = self.actor(state)
        mean, std = torch.chunk(actor_output, 2, dim=-1)
        std = F.softplus(std) + 1e-5  # 确保标准差为正
        dist = D.Normal(mean, std)
        value = self.critic(state)
        return dist, value


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class LSHADE_PPO:
    def __init__(self, fitness, constraints, lower, upper, pop_size, dim, epochs,
                 gamma, learning_rate, clip_epsilon, n_epochs, batch_size):
        self.fitness = fitness
        self.constraints = constraints
        self.lower = lower
        self.upper = upper
        self.pop_size = pop_size
        self.dim = dim
        self.epochs = epochs
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.population = np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))
        self.fit = np.array([self.fitness(ind) for ind in self.population])
        self.conv = np.array([self.constraints(ind) for ind in self.population])

        self.scaler = StandardScaler()
        self.scaler.fit(self.population)

        self.ppo = PPONetwork(input_dim=self.dim, output_dim=2).to(self.device)
        self.optimizer = optim.Adam(self.ppo.parameters(), lr=self.learning_rate)
        self.memory = PPOMemory(self.batch_size)

        self.true_pareto_front = self.load_true_pareto_front()
        self.pareto_front = self.calculate_pareto_front()

        self.hypervolume_history = []
        self.loss_history = []
        self.reward_history = []
        self.cr_history = []
        self.f_history = []

        # 设置实时绘图
        plt.ion()
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('LSHADE-PPO 训练过程')

    def load_true_pareto_front(self):
        if os.path.exists('true_pareto_front.mat'):
            data = loadmat('true_pareto_front.mat')
            return data['PF']
        else:
            print("真实Pareto前沿文件不存在，初始化为空集合。")
            return np.empty((0, 2))

    def calculate_pareto_front(self):
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
        if pareto_front.size == 0:
            return 0.0
        sorted_pf = pareto_front[pareto_front[:, 0].argsort()]
        hypervolume = 0.0
        previous_f1 = 0.0
        for f1, f2 in sorted_pf:
            hypervolume += (f2) * (f1 - previous_f1)
            previous_f1 = f1
        hypervolume += (reference_point[0] - previous_f1) * reference_point[1]
        return hypervolume

    def calculate_diversity_reward(self, population):
        pareto = self.calculate_pareto_front()
        hypervolume = self.calculate_hypervolume(pareto)
        return hypervolume

    def is_dominated(self, new_fit, current_fit):
        return np.all(new_fit <= current_fit) and np.any(new_fit < current_fit)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        dist, value = self.ppo(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = action.squeeze().detach().cpu().numpy()
        value = value.squeeze().item()
        return action, log_prob.item(), value

    def mutate(self, idx, F):
        idxs = list(range(0, self.pop_size))
        idxs.remove(idx)
        a, b, c = random.sample(idxs, 3)
        mutant = self.population[a] + F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower, self.upper)
        return mutant

    def crossover(self, parent, mutant, CR):
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, parent)
        return trial

    def select(self, idx, trial, new_fit, new_conv):
        if new_conv <= self.conv[idx]:
            if self.is_dominated(new_fit, self.fit[idx]):
                self.population[idx] = trial
                self.fit[idx] = new_fit
                self.conv[idx] = new_conv

    def update_pareto_front(self):
        self.pareto_front = self.calculate_pareto_front()

    def learn(self):
        total_loss = 0
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * 0.95
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)

            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                dist, critic_value = self.ppo(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions).sum(dim=-1)
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * \
                                         advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                loss = actor_loss + 0.5 * critic_loss
                total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory.clear_memory()
        return total_loss / (self.n_epochs * len(batches))

    def update_plots(self, epoch):
        # 更新个体分布图
        self.axs[0, 0].clear()
        self.axs[0, 0].scatter(self.fit[:, 0], self.fit[:, 1], alpha=0.6)
        if self.true_pareto_front.size > 0:
            self.axs[0, 0].scatter(self.true_pareto_front[:, 0], self.true_pareto_front[:, 1],
                                   color='red', label='真实Pareto前沿')
        self.axs[0, 0].set_xlabel('f1')
        self.axs[0, 0].set_ylabel('f2')
        self.axs[0, 0].set_title('个体分布')
        self.axs[0, 0].legend()

        # 更新损失图
        self.axs[0, 1].clear()
        self.axs[0, 1].plot(self.loss_history)
        self.axs[0, 1].set_xlabel('Epoch')
        self.axs[0, 1].set_ylabel('Loss')
        self.axs[0, 1].set_title('训练损失')

        # 更新奖励图
        self.axs[1, 0].clear()
        self.axs[1, 0].plot(self.reward_history)
        self.axs[1, 0].set_xlabel('Epoch')
        self.axs[1, 0].set_ylabel('Reward')
        self.axs[1, 0].set_title('奖励')

        # 更新CR和F参数图
        self.axs[1, 1].clear()
        self.axs[1, 1].plot(self.cr_history, label='CR')
        self.axs[1, 1].plot(self.f_history, label='F')
        self.axs[1, 1].set_xlabel('Epoch')
        self.axs[1, 1].set_ylabel('Value')
        self.axs[1, 1].set_title('CR和F参数')
        self.axs[1, 1].legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def run(self):
        for epoch in range(1, self.epochs + 1):
            epoch_reward = 0
            scaled_population = self.scaler.transform(self.population)

            cr_sum, f_sum = 0, 0
            for i in range(self.pop_size):
                state = scaled_population[i]
                action, prob, val = self.choose_action(state)
                F, CR = np.clip(action, 0, 1)  # 确保F和CR在[0,1]范围内
                cr_sum += CR
                f_sum += F
                mutant = self.mutate(i, F)
                trial = self.crossover(self.population[i], mutant, CR)
                new_fit = self.fitness(trial)
                new_conv = self.constraints(trial)
                self.select(i, trial, new_fit, new_conv)

                done = False
                reward = self.calculate_diversity_reward(self.population)
                epoch_reward += reward
                self.memory.store_memory(state, action, prob, val, reward, done)

            self.update_pareto_front()
            hypervolume = self.calculate_hypervolume(self.pareto_front)
            self.hypervolume_history.append(hypervolume)

            if len(self.memory.states) >= self.batch_size:
                loss = self.learn()
                self.loss_history.append(loss)

            avg_reward = epoch_reward / self.pop_size
            self.reward_history.append(avg_reward)
            self.cr_history.append(cr_sum / self.pop_size)
            self.f_history.append(f_sum / self.pop_size)

            print(f'第 {epoch} 轮, 超体积: {hypervolume:.4f}, 损失: {loss:.4f}, 平均奖励: {avg_reward:.4f}')

            self.update_plots(epoch)

        torch.save(self.ppo.state_dict(), 'ppo_model.pth')
        print("模型已保存到: ppo_model.pth")

    def show_final_result(self):
        plt.ioff()
        plt.figure(figsize=(10, 6))
        if self.true_pareto_front.size > 0:
            plt.scatter(self.true_pareto_front[:, 0], self.true_pareto_front[:, 1],
                        color='green', label='真实 Pareto 前沿', marker='x')
        if self.pareto_front.size > 0 and self.pareto_front.ndim == 2:
            plt.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1],
                        color='blue', label='获得的 Pareto 前沿', alpha=0.6)
        if self.fit.ndim == 2:
            plt.scatter(self.fit[:, 0], self.fit[:, 1],
                        color='red', label='所有个体', alpha=0.3)
        plt.xlabel('目标 1 (f1)')
        plt.ylabel('目标 2 (f2)')
        plt.title('最终 Pareto 前沿和种群适应度')
        plt.legend()
        plt.grid(True)
        plt.show()


def fitness(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - sqrt(f1 / g))
    return np.array([f1, f2])


def constraints(x):
    return 0.0


if __name__ == "__main__":
    lshade_ppo = LSHADE_PPO(
        fitness=fitness,
        constraints=constraints,
        lower=0.0,
        upper=1.0,
        pop_size=100,
        dim=10,
        epochs=100,
        gamma=0.99,
        learning_rate=0.0003,
        clip_epsilon=0.2,
        n_epochs=10,
        batch_size=64
    )

    lshade_ppo.run()
    lshade_ppo.show_final_result()