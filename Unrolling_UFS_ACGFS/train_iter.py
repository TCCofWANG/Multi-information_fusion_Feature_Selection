import argparse
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from model.ACGFS import ACGFS
from utils import *
from L_metrix import *
from datetime import datetime
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='PyTorch DARE Training',add_help=True)
parser.add_argument('--T', default=70, type=int, help='number of iteration')
parser.add_argument('--dk', default=20, type=int, help='hidden layer num [10,20,```,50]')
parser.add_argument('--k', default=5, type=int, help='nearest neighbor nums')
parser.add_argument('--sigma', default=1000.0, type=float)
parser.add_argument('--gm', default=10.0, type=float, help='regular of sparse FS')
parser.add_argument('--beta', default=0.01, type=float, help='regular weight sparse of W_1')
parser.add_argument('--nd', default=0.01, type=float, help='regular of sparse W')
parser.add_argument('--ro', default=1e-2, type=float, help='step size of φ')
parser.add_argument('--miu', default=1e-2, type=float, help='step size of N')
parser.add_argument('--eta', default=1e-4, type=float, help='step size of W')
parser.add_argument('--P', default=50, type=int, help='select feature number [10, 20, ```,150]')
parser.add_argument('--c', default=15, type=int, help='channel of k cluster')
parser.add_argument('--datasets', default='Yale', type=str, help='dataset of training')
parser.add_argument('--kmeans', default=15, type=int, help='test k-means cluster num')

args = parser.parse_args()
"123, 132"
seed = 132
# seed = 123
def tanh(x):
    exp = np.exp(x)
    exp_x = np.exp(-x)

    #防止溢出
    inf_indices = np.isinf(exp_x)
    exp_x[inf_indices] = np.finfo(float).max
    inf_indices = np.isinf(exp)
    exp[inf_indices] = np.finfo(float).max

    # 计算 tanh 的分子和分母
    numerator = exp - exp_x
    denominator = exp + exp_x

    # 处理分母为 0 的情况，避免除零错误
    denominator[denominator == 0] = np.finfo(float).eps

    # 计算 tanh 函数值
    tanh = numerator / denominator
    return tanh

class DataLoader:
    def __init__(self, dataset, batch_size=10, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        batch = []
        for idx in self.indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield tuple(batch)
                batch = []
        if len(batch) > 0:
            yield tuple(batch)


def normlization(x, mean, std):
    x = x / 255.0
    output = (x - mean) / std

    return output


def cluster_acc(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    _make_cost_m = lambda x:-x + np.max(x)
    indexes = linear_sum_assignment(_make_cost_m(cm))
    indexes = np.concatenate([indexes[0][:,np.newaxis],indexes[1][:,np.newaxis]], axis=-1)
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    acc = np.trace(cm2) / np.sum(cm2)
    return acc

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x:torch.flatten(x))
])
# transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),  # 转为灰度图
#         transforms.ToTensor(),  # 转为张量
#         transforms.Normalize((0.5,), (0.5)),
#         transforms.Lambda(lambda x: torch.flatten(x))  # 转为向量
#     ])
# # train_dataset = torchvision.datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform)
# test_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
# # train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
# test_data_iter = iter(test_loader)
# test_fea, test_labels = next(test_data_iter)
# features = test_fea.numpy()
# labels = test_labels.numpy()
# fea_num = features.shape[1]
# sam_num = features.shape[0]
# sparse_matrix_test = load_npz("./graph_matrix/sparse_lap_test_Cifar10.npz")
# L = sparse_matrix_test.toarray()
# setup_seed(seed)
# transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),  # 转为灰度图
#         transforms.ToTensor(),  # 转为张量
#         transforms.Lambda(lambda x: torch.flatten(x))  # 转为向量
#     ])
# # train_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
# test_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
#
# # 定义数据加载器
# # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
#
# # 获取灰度化并向量化的训练数据和标签
# # train_data_iter = iter(train_loader)
# # train_images, train_labels = next(train_data_iter)
#
# # 获取灰度化并向量化的测试数据和标签
# test_data_iter = iter(test_loader)
# test_images, test_labels = next(test_data_iter)
# features = test_images.numpy()
# labels = test_labels.numpy()
# fea_num = features.shape[1]
# sam_num = features.shape[0]
data = loadmat(('./datasets/' + args.datasets + '.mat'))
setup_seed(seed)
features = data['X'].astype(float)
labels = data['Y']
scaler = StandardScaler()
features = scaler.fit_transform(features)
labels = labels.reshape(-1,)
#
# best_acc = 0
# best_nmi = 0
fea_num, sam_num = features.shape[1], features.shape[0]
w = np.random.randn(fea_num, args.dk)
d = np.eye(fea_num)
fi = np.eye(args.c)
I = np.eye(args.c)
n = np.random.rand(args.c, sam_num)
A = compute_knn_gaussian_adjacency_matrix(features, args.k, args.sigma)
L = compute_laplacian_matrix(A)
m = np.random.rand(args.dk, args.c)

epoch_loss = []
t0 = time.time()
for i in range(args.T):
    w, d, m, n, fi = ACGFS(features.T, w, d, m, n, fi, I, args.nd, args.beta, args.gm, L, args.eta, args.miu, args.ro)
    loss = ((np.linalg.norm(np.matmul(w, tanh(np.matmul(w.T, features.T))) - features.T, ord='fro'))
        + args.beta * np.linalg.norm(tanh(np.matmul(w.T, features.T)-np.matmul(m, n)) + args.gm * np.trace(np.matmul(np.matmul(n, L), n.T))))
    epoch_loss.append(loss)
    print("====> {} Train step finished.".format(i))
# print(np.max(w))
print("====> testing")
t1 = time.time()
norm_w = np.linalg.norm(w, ord=2, axis=1)
topk_indices = np.argsort(norm_w)[-args.P:]
ufs_fea = np.zeros((sam_num, args.P))
for i in range(sam_num):
    ufs_fea[i, range(args.P)] = features[i, topk_indices]
accuracy_t = []
nmi_t = []
for _ in range(20):
    kmeans = KMeans(args.kmeans, random_state=_)
    kmeans.fit(ufs_fea)
    cluster_labels = kmeans.labels_

    accuracy_t.append(cluster_acc(labels, cluster_labels))
    nmi_t.append(normalized_mutual_info_score(labels, cluster_labels, average_method='max'))

print("===>Accuracy: {:.4f}%".format(np.mean(accuracy_t) * 100))
print("===>acc_std: {:.4f}%".format(np.std(accuracy_t) * 100))
print('')
print("===>NMI: {:.4f}".format(np.mean(nmi_t)*100))
print("===>nmi_std: {:.4f}%".format(np.std(nmi_t)*100))
print('Spent time: {}'.format((t1-t0)))
# plt.figure(figsize=(10, 6))
# plt.plot(range(1,args.T+1), epoch_loss, label='ACGFS')
# # plt.title('ACGFS')
# plt.xlabel('Iteration Number', fontsize=20)
# plt.ylabel('Object function value', fontsize=20)
# plt.legend(fontsize=20)
# plt.grid(True)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-3, 3))  # 设置在何种范围内使用科学计数法
# plt.gca().yaxis.set_major_formatter(formatter)
# plt.show()
# if not os.path.isdir('results/'):
#     os.mkdir('results/')
#
# log_path = './results/UFS_model4_COIL20_999.csv'
# if not os.path.exists(log_path):
#     table_head = [['dataset', 'algo','hidden_layer_k', 'K_near', 'sigma', 'nd', 'beta', 'gm',
#                  'feature_num', 'cluster_num', 'iter_num',
#                  'mean_acc', 'std_acc', 'mean_nmi', 'std_nmi']]
#     write_csv(log_path, table_head, 'w+')
#
# time_p = datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间
# a_log = [{'dataset': args.datasets, 'algo': 'model_4',
#         'hidden_layer_k': args.dk,'K_near': args.k,'sigma':args.sigma,'nd':args.nd,'beta':args.beta,'gm':args.gm,
#         'feature_num': args.P,'cluster_num': args.c, 'iter_num': args.T,
#         'mean_acc': np.mean(accuracy_t) * 100, 'std_acc': np.std(accuracy_t) * 100,
#         'mean_nmi': np.mean(nmi_t)*100, 'std_nmi':np.std(nmi_t)*100}]
# write_csv_dict(log_path, a_log, 'a+')




