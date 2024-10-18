import numpy as np
from sklearn.neighbors import NearestNeighbors


# def gaussian_kernel(x, y, sigma=1.0):
#     """计算两个向量之间的高斯核相似度"""
#     return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

# def compute_similarity_matrix(data, sigma=1.0):
#     """计算数据矩阵的相似度矩阵"""
#     n = data.shape[0]
#     similarity_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             similarity_matrix[i, j] = gaussian_kernel(data[i, :], data[j, :], sigma)
#     return similarity_matrix
#
# def knn_graph(similarity_matrix, k):
#     """基于相似度矩阵构建k近邻图的邻接矩阵"""
#     n = similarity_matrix.shape[1]
#     adj_matrix = np.zeros((n, n))
#     for i in range(n):
#         knn_indices = np.argsort(similarity_matrix[i])[-(k+1):-1]  # 取k个近邻，排除自己
#         for j in knn_indices:
#             adj_matrix[i, j] = similarity_matrix[i, j]
#             adj_matrix[j, i] = similarity_matrix[i, j]  # 无向图
#     for j in range(n):
#         knn_indices = np.argsort(similarity_matrix[:, j])[-(k+1):-1]  # 取k个近邻，排除自己
#         for i in knn_indices:
#             adj_matrix[i, j] = similarity_matrix[i, j]
#             adj_matrix[j, i] = similarity_matrix[i, j]  # 无向图
#     return adj_matrix

def compute_laplacian_matrix(adj_matrix):
    """计算无归一化的图拉普拉斯矩阵"""
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian_matrix = degree_matrix - adj_matrix
    return laplacian_matrix


def gaussian_kernel(distances, sigma=1.0):

    return np.exp(-distances / (2*sigma**2))

def distance_oshi(x):
    x_norm_squared = np.sum(x ** 2, axis=1)

    # 计算y中所有点的平方和，沿着第二个维度（列）
    y_norm_squared = np.sum(x ** 2, axis=1)

    # 使用广播计算x和y中所有点之间的欧几里得距离的平方
    distances_squared = -2 * x.dot(x.T) + x_norm_squared[:, np.newaxis] + y_norm_squared[np.newaxis, :]

    # 开方得到距离矩阵
    distances = distances_squared
    return distances

def compute_knn_gaussian_adjacency_matrix(data, k, sigma=100.0):
    # 计算欧氏距离矩阵
    distances = distance_oshi(data)
    # 转换为高斯相似度矩阵
    gaussian_similarities = gaussian_kernel(distances - np.diag(distances), sigma)

    # 找到每个样本的 k 近邻
    # data_np = data.numpy()
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
    _, indices = nbrs.kneighbors(data)

    # 初始化邻接矩阵
    n_samples = data.shape[0]
    adjacency_matrix = np.zeros((n_samples, n_samples))

    # 遍历每个样本
    for i in range(n_samples):
        for j in indices[i]:
            if i != j:  # 不包含自己
                adjacency_matrix[i, j] = gaussian_similarities[i, j]

    return adjacency_matrix