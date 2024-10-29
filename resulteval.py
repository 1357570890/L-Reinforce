from math import cos, pi, sin
import numpy as np

def evaluate_individual(individual, evaluation_functions):
    """
    评估个体的适应度值。

    参数:
    individual: 需要评估的个体数据（例如最优个体值）。
    evaluation_functions: 用于评估个体的函数列表。

    返回:
    评估结果（适应度值和约束值）。
    """
    fitness_values = [func(individual) for func in evaluation_functions]
    return fitness_values

# 示例评估函数
def fitness(x):
    f1 = sum(xi ** 2 - 10 * cos(2 * pi * xi) + 10 for xi in x)  # 目标1
    f2 = sum((xi - 5) ** 2 for xi in x)  # 目标2
    return np.array([f1, f2])  # 返回多个目标

def constraints(x):
    g1 = sum(-xi * sin(2 * xi) for xi in x)
    g2 = sum(xi * sin(xi) for xi in x)
    return max(g1, 0) + max(g2, 0)

def format_individual(individual_str):
    """
    将个体字符串转换为数组。

    参数:
    individual_str: 以空格分隔的个体字符串。

    返回:
    转换后的数组。
    """
    # 去掉方括号并分割字符串，然后转换为浮点数数组
    individual_list = individual_str.strip('[]').split()
    return np.array([float(x) for x in individual_list])

# 示例个体数据（字符串形式）
individual_str = '[ 4.03020096  4.160532   -2.06032085  1.02609444  3.99100304  3.0364666 0.02560377  2.03740191  3.04575729  1.02470779]'
# 将字符串转换为数组
best_individual = format_individual(individual_str)

# 评估个体
evaluation_functions = [fitness, constraints]
results = evaluate_individual(best_individual, evaluation_functions)

# 打印评估结果
print("适应度评估结果:", results[0])  # 适应度值
print("约束评估结果:", results[1])    # 约束值
