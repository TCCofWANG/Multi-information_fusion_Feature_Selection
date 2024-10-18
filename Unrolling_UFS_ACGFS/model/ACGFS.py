import numpy as np

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

def relu(x):
    return np.maximum(0, x)

def mm(x, y):
    return np.matmul(x, y)


def ACGFS(x, w, d, m, n, fi, I, nd, beta, gama, L, eta, miu, ro):
    f_x = tanh(np.matmul(w.T, x))
    f_x_ = 1 - f_x ** 2

    ntn_inv = np.linalg.inv(mm(n, n.T)+I)
    nm_t = mm(f_x, n.T)
    nm = mm(nm_t, ntn_inv)

    n_grad = -beta * mm(m.T, f_x) + beta * mm(mm(m.T, m), n) + gama * mm(n, L) - mm(fi, n)
    nn = n - miu * n_grad
    nn = relu(nn)
    n_fi = fi - ro * (mm(nn, nn.T)-I)

    t1 = x - np.matmul(w, f_x)
    t2 = f_x + np.matmul(w.T, x) * f_x_
    t3 = f_x - np.matmul(m, n)
    t4 = beta * np.matmul(x, (t3 * f_x_).T)
    grad = np.matmul(t1, t2.T) - t4
    nw = w - eta * (nd * np.matmul(d, w) - grad)

    d = np.diag(np.linalg.norm(nw, ord=2, axis=1))

    return nw, d, nm, nn, n_fi