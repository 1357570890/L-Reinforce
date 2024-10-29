import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# 定义个体类
class Individual:
    def __init__(self, position, label, centroids, data, labels):
        self.position = position
        self.label = label
        self.cluster_center = centroids[label]  # 选取到的聚类中心
        self.distance_to_center = np.linalg.norm(position - self.cluster_center)  # 与聚类中心的距离
        self.rank = self.calculate_rank(self.distance_to_center, centroids, data, labels)  # 计算等级

    def calculate_rank(self, distance_to_center, centroids, data, labels):
        # 获取同一聚类中的个体
        cluster_data = data[labels == self.label]

        # 计算同一聚类中的个体与其聚类中心的距离
        distances = np.linalg.norm(cluster_data - centroids[self.label], axis=1)

        # 计算最小和最大距离
        min_distance = np.min(distances)
        max_distance = np.max(distances)

        # 根据距离分档次
        if min_distance == max_distance:
            return 1  # 所有个体在同一位置
        else:
            # 归一化个体与聚类中心的距离
            normalized_distance = (distance_to_center - min_distance) / (max_distance - min_distance)

            # 使用normalized_distance来判断等级
            if normalized_distance < 0.33:
                return 1  # 高等级
            elif normalized_distance < 0.66:
                return 2  # 中等级
            else:
                return 3  # 低等级

# 生成随机数据
np.random.seed(42)

# 设置样本数量和特征维数
num_samples = 300
num_features = 3  # 固定为3维

# 生成三个聚类的数据
cluster_centers = np.random.rand(3, num_features) * 10  # 随机生成三个聚类中心
data = []

for center in cluster_centers:
    # 为每个聚类中心生成随机数据
    cluster_data = np.random.randn(num_samples // 3, num_features) + center
    data.append(cluster_data)

# 合并数据
data = np.vstack(data)

# KMeans聚类
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)

# 获取聚类结果
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 创建个体对象
individuals = [Individual(data[i], labels[i], centroids, data, labels) for i in range(data.shape[0])]

# 输出每个个体的位置信息、等级、选取的聚类中心和与聚类中心的距离
for i, individual in enumerate(individuals):
    print(f"个体 {i}: 位置 {individual.position}, 等级 {individual.rank}, "
          f"选取的聚类中心 {individual.cluster_center}, 与聚类中心的距离 {individual.distance_to_center:.2f}")

# 可视化聚类结果（三维展示图）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=30, cmap='viridis')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=200, alpha=0.75, marker='X')
ax.set_title("KMeans Clustering Result")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
plt.show()
