import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from skfeature.utility.construct_W import construct_W
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse import load_npz
from L_metrix import compute_knn_gaussian_adjacency_matrix, compute_laplacian_matrix
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5)),
    transforms.Lambda(lambda x:torch.flatten(x))
])
train_dataset = torchvision.datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
train_data_iter = iter(train_loader)
test_data_iter = iter(test_loader)
train_fea, train_labels = next(train_data_iter)
test_fea, test_labels = next(test_data_iter)
# features = test_fea.numpy()
# labels = test_labels.numpy()
# fea_num = features.shape[1]
# sam_num = features.shape[0]
# transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),  # 转为灰度图
#         transforms.ToTensor(),  # 转为张量
#         transforms.Lambda(lambda x: torch.flatten(x))  # 转为向量
#     ])
# train_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
# test_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
#
# # 定义数据加载器
# train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
#
# # 获取灰度化并向量化的训练数据和标签
# train_data_iter = iter(train_loader)
# train_images, train_labels = next(train_data_iter)
#
# # 获取灰度化并向量化的测试数据和标签
# test_data_iter = iter(test_loader)
# test_images, test_labels = next(test_data_iter)
kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1.0}
W = construct_W(train_fea.numpy(), **kwargs)

W = (W + W.T) / 2
W_norm = np.diag(np.sqrt(1 / W.sum(1)))
W = np.dot(W_norm, np.dot(W, W_norm))
WT = W.T
W[W < WT] = WT[W < WT]

S = construct_W(test_fea.numpy(), **kwargs)

S = (S + S.T) / 2
S_norm = np.diag(np.sqrt(1 / S.sum(1)))
S = np.dot(S_norm, np.dot(S, S_norm))
ST = S.T
S[S < ST] = ST[S < ST]
L_test = np.eye(S.shape[0]) - S
L_train = np.eye(W.shape[0]) - W
#
# # 转换为稀疏矩阵格式
sparse_L_train = csr_matrix(L_train)
sparse_L_test = csr_matrix(L_test)
#
#
# # 保存稀疏矩阵
save_npz("sparse_lap_train_Cifar100.npz", sparse_L_train)
save_npz("sparse_lap_test_Cifar100.npz", sparse_L_test)


#加载图矩阵
# sparse_matrix_train = load_npz("sparse_lap_train_cifar10.npz")
# sparse_matrix_test = load_npz("sparse_lap_test_cifar10.npz")
#
# L_train = sparse_matrix_train.toarray()
# L_test = sparse_matrix_test.toarray()
# print(L_test)
# print(L_train)

# fea_num = train_images.size(1)
# A_train = compute_knn_gaussian_adjacency_matrix(train_images.numpy(), 5, 1000.0)
# L_train = compute_laplacian_matrix(A_train)
# A_test = compute_knn_gaussian_adjacency_matrix(test_images.numpy(), 5, 1000.0)
# L_test = compute_laplacian_matrix(A_test)
#
# matrix_train = L_train
# matrix_test = L_test
# torch.save({'matrix1': matrix_train, 'matrix2': matrix_test}, 'matrices_L_cifar10.pt')
# print("Matrices saved successfully.")

# metrix = torch.load('matrices_L_cifar10.pt')
# w = metrix['matrix1']
# m = metrix['matrix2']
#
# print(w)
# print(m)