import argparse
import os
import time
import torch
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from L_metrix import compute_knn_gaussian_adjacency_matrix, compute_laplacian_matrix
from model.UACGFS import UACGFS
from utils import setup_seed

batch_size = 64
epoches = 10

parser = argparse.ArgumentParser(description='PyTorch UFS Training', add_help=True)
parser.add_argument('--cudaID', default='0', type=str, help='main cuda ID')
parser.add_argument('--cluster_num', default=10, type=int, help='k-means cluster numbers')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--start_seed', default=1, type=int, help='seed')
parser.add_argument('--e', default=1e-6, type=float, help='small positive constant')
parser.add_argument('--P', default=250, type=int, help='select feature number')
parser.add_argument('--k', default=10, type=int, help='hidden layer of autoencoder')
parser.add_argument('--c', default=10, help='cluster channels of autoencoder')
parser.add_argument('--T', default=4, type=int, help='iteration num')
parser.add_argument('--knn', default=5, type=int, help='nearest neighborhood num')
parser.add_argument('--sigma', default=1000.0, type=float, help='gaussa')
parser.add_argument('--model_name', default='ACGFS', type=str, help='type of model')
parser.add_argument('--save_folder', default='./checkpoints', type=str)
parser.add_argument('--statistic', default='./results', type=str)
parser.add_argument('--data_name', default='USPS', type=str, help='type of data')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cudaID
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cluster_acc(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    _make_cost_m = lambda x:-x + np.max(x)
    indexes = linear_sum_assignment(_make_cost_m(cm))
    indexes = np.concatenate([indexes[0][:,np.newaxis],indexes[1][:,np.newaxis]], axis=-1)
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    acc = np.trace(cm2) / np.sum(cm2)
    return acc


def custom_collate_fn(batch):
    """
    自定义 collate_fn，将每个批次的数据、标签和图矩阵拼接在一起

    参数:
    batch (list): 包含多个样本的列表，每个样本是一个字典，包含数据、标签和图矩阵

    返回:
    dict: 包含数据、标签和图矩阵的字典
    """
    data = torch.stack([item['data'] for item in batch])  # Shape: [c, d]
    labels = torch.tensor([item['label'] for item in batch])  # Shape: [c]

    # 拼接图矩阵，形成一个 batch_size x batch_size 的大矩阵
    c = len(batch)
    indices = torch.tensor([item['index'] for item in batch])
    graph_matrix = torch.zeros((c, c))
    for i in range(c):
        graph_matrix[:, i] = batch[i]['graph_matrix'][indices,]
    return {data, labels, graph_matrix}


class CIFAR10WithGraphs(Dataset):
    def __init__(self, image_data, graph_data, transform=None):
        self.image_data = image_data
        self.graph_data = graph_data
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image, label = self.image_data[idx]

        # 加载对应的图矩阵
        graph_matrix = self.graph_data[idx]

        if self.transform:
            image = self.transform(image)

        # 返回图像、图矩阵和标签
        return image, graph_matrix, label


class CustomDataset(Dataset):
    def __init__(self, data, labels, graph_matrices):
        """
        初始化数据集

        参数:
        data (Tensor): 样本数据，形状为 [n, d]
        labels (Tensor): 样本标签，形状为 [n]
        graph_matrices (Tensor): 图矩阵，形状为 [n, n]
        """
        self.data = data
        self.labels = labels
        self.graph_matrices = graph_matrices
        self.indices = torch.arange(len(data))

    def __len__(self):
        # 返回数据集的样本数量
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取样本数据、标签和对应的子图矩阵

        参数:
        idx (int): 样本索引

        返回:
        dict: 包含数据、标签和图矩阵的字典
        """
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx],
            'graph_matrix': self.graph_matrices[idx],
            'index': self.indices[idx]
        }
        return sample

class GraphDataset(Dataset):
    def __init__(self, data, labels, graph_matrix):
        """
        data: 输入数据（样本矩阵），大小为 (1000, 1024)
        labels: 样本标签，大小为 (1000,)
        graph_matrix: 图矩阵，大小为 (1000, 1000)
        """
        self.data = data
        self.labels = labels
        self.graph_matrix = graph_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取第 idx 个样本、标签及其对应的图相似性矩阵行
        sample = self.data[idx]
        label = self.labels[idx]
        graph_row = self.graph_matrix[idx]

        return sample, label, graph_row




def feature_select(w1, x, P):
    x = x.float()
    sam_num = x.shape[1]
    W_1_norms = torch.norm(w1, dim=1).reshape(-1, 1)
    top_P_values, top_P_indices = torch.topk(W_1_norms, P, dim=0)
    top_P_indices = top_P_indices.reshape(-1, )
    fs_metric = torch.zeros((sam_num, P))
    for i in range(sam_num):
        # test1 = torch.arange(P)

        fs_metric[i, torch.arange(P)] = x[top_P_indices, i]

    outputs = fs_metric

    return outputs


def train(epoch):
    epoch_loss = 0

    for t, (input, label, l) in enumerate(train_loader):
        x = input.t().to(device)
        x = x.float()
        if l.ndimension() == 1:
            temp_lab = l
            l = label
            label = temp_lab
        l = l.float().to(device)

        model.zero_grad()
        optimizer.zero_grad()
        t0 = time.time()
        _, loss = model(x, l)
        t1 = time.time()

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch + 1,
                                                                                 t + 1, len(train_loader), loss.data,
                                                                                 (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f} ".format(epoch + 1,epoch_loss / len(train_loader)))
    return epoch_loss


def test():
    model.eval()
    weight = torch.zeros((fea_num, args.k)).to(device)
    t0 = time.time()
    for t, (input, label, l) in enumerate(test_loader):
        x = input.t().to(device)
        x = x.float()
        if l.ndimension() == 1:
            temp_lab = l
            l = label
            label = temp_lab
        l = l.float().to(device)
        with torch.no_grad():
            w, _ = model(x, l)
            weight += w
    t1 = time.time()
    fea_weight = weight / len(test_loader)
    fea_weight = fea_weight.cpu()

    acc_best = 0
    ufs_fea = feature_select(fea_weight, fea_test.t(), args.P)
    accuracy_t = []
    nmi_t = []
    for kseed in range(20):
        kmeans = KMeans(args.cluster_num, random_state=kseed)
        kmeans.fit(ufs_fea.cpu().numpy())
        cluster_labels = kmeans.labels_

        accuracy_t.append(cluster_acc(labels_test.numpy(), cluster_labels))
        nmi_t.append(normalized_mutual_info_score(labels_test.numpy(), cluster_labels, average_method='max'))

    print("===>Accuracy: {:.4f}%".format(np.mean(accuracy_t) * 100))
    print("===>acc_std: {:.4f}%".format(np.std(accuracy_t) * 100))
    print('')
    print("===>NMI: {:.4f}".format(np.mean(nmi_t) * 100))
    print("===>nmi_std: {:.4f}%".format(np.std(nmi_t) * 100))
    print('Spent time: {}'.format((t1-t0)))
    return fea_weight


if __name__ == '__main__':
    print('===> Loading datasets')
    setup_seed(121)

    data = loadmat('./datasets/' + args.data_name)
    features = data['fea']
    labels = data['gnd']
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = torch.from_numpy(features)
    fea_num = features.shape[1]
    labels = torch.from_numpy(labels.reshape(-1,))
    fea_train, fea_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.8,
                                                                      random_state=90)
    # labels_test = labels_test.reshape(-1, )
    # labels_train = labels_train.reshape(-1, )
    A_train = compute_knn_gaussian_adjacency_matrix(fea_train.numpy(), args.knn, args.sigma)
    L_train = torch.from_numpy(compute_laplacian_matrix(A_train))
    A_test = compute_knn_gaussian_adjacency_matrix(fea_test.numpy(), args.knn, args.sigma)
    L_test = torch.from_numpy(compute_laplacian_matrix(A_test))

    # train_set = torch.utils.data.TensorDataset(fea_train, labels_train)
    # test_set = torch.utils.data.TensorDataset(fea_test, labels_test)
    train_set = CustomDataset(fea_train, labels_train, L_train)
    test_set = CustomDataset(fea_test, labels_test, L_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=custom_collate_fn)

    """FMNIST数据加载"""
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,)),
    #     transforms.Lambda(lambda x: torch.flatten(x))
    # ])
    # train_dataset = torchvision.datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform)
    # test_dataset = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    # train_data_iter = iter(train_loader)
    # test_data_iter = iter(test_loader)
    # train_fea, train_labels = next(train_data_iter)
    # test_fea, test_labels = next(test_data_iter)
    # fea_num = train_fea.size(1)
    # sparse_matrix_train = load_npz("sparse_lap_train_Cifar100.npz")
    # sparse_matrix_test = load_npz("sparse_lap_test_Cifar100.npz")
    # L_train = torch.from_numpy(sparse_matrix_train.toarray())
    # L_test = torch.from_numpy(sparse_matrix_test.toarray())
    #
    # train_set = CustomDataset(train_fea, train_labels, L_train)
    # test_set = CustomDataset(test_fea, test_labels, L_test)
    #
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
    #                                            collate_fn=custom_collate_fn)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False,
    #                                           collate_fn=custom_collate_fn)
    print('==> Building model..')
    model = UACGFS(fea_num, args.k, args.c)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    # std_acc_b = 0
    # best_nmi = 0
    # std_nmi_b = 0
    for epoch in range(epoches):
        _ = train(epoch)
        __ = test()

