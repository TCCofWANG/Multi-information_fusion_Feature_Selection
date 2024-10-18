import numpy as np
import torch
import torch.nn as nn

class Res_Net(nn.Module):
    def __init__(self):
        super(Res_Net, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 1, 3, 1, 1)
        self.ac = nn.Tanh()
        self.conv2d2 = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x):
        input = x.unsqueeze(0).unsqueeze(0)
        out = self.conv2d1(input)
        out = self.ac(out)
        output = self.conv2d2(out)

        output = output.squeeze(0).squeeze(0)
        return output


class Net_D(nn.Module):
    def __init__(self, d, k):
        super(Net_D, self).__init__()
        self.fc = nn.Linear(d*k, d*k)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        input = x.view(x.size(0)*x.size(1),)
        out = self.fc(input)
        out = self.dropout(out)
        out = out.view(x.size(0), x.size(1))

        output = out + x
        return output

class Head(nn.Module):
    def __init__(self, d, k, c):
        super(Head, self).__init__()
        self.d = d
        self.k = k
        self.c = c
        self.I = torch.eye(c).cuda()
        self.fi = torch.eye(c).cuda()
        self.w = torch.randn(d, k).cuda()

    def forward(self, x):
        I = self.I
        n = torch.rand(self.c, x.size(1)).cuda()
        fi = self.fi
        w = self.w

        return w, fi, n, I

def init_compute(x, w):
    f_x = torch.tanh(torch.matmul(w.t(), x))
    f_x_grad = 1 - f_x ** 2
    return f_x, f_x_grad

def Grad_N(f_x, m, n, l, fi, gama, beta):
    t1 = -2 * beta * torch.matmul(m.t(), f_x)
    t2 = 2*beta*torch.matmul(torch.matmul(m.t(), m), n)
    t3 = 2 * gama * torch.matmul(n, l)
    t4 = -2 * torch.matmul(fi, n)
    grad = t1 + t2 + t3 + t4

    return grad

def Grad_fi(n):
    I = torch.eye(n.size(0)).cuda()
    grad = torch.matmul(n, n.t()) - I
    return grad

def Grad_W(x,f_x,f_x_grad, w, m, n, beta):
    t1 = (f_x - torch.matmul(m,n)) * f_x_grad
    t2 = beta * torch.matmul(x, t1.t())
    t3 = x - torch.matmul(w, f_x)
    t4 = f_x - torch.matmul(w.t(), x) * f_x_grad
    t5 = torch.matmul(-t3, t4.t())
    grad = 2*t2 + 2*t5
    return grad

class Net_inv(nn.Module):
    def __init__(self):
        super(Net_inv, self).__init__()
        self.alpha1 = Res_Net()
        self.beta1 = Res_Net()

        self.alpha2 = Res_Net()
        self.beta2 = Res_Net()

        self.alpha3 = Res_Net()
        self.beta3 = Res_Net()

        self.alpha4 = Res_Net()
        self.beta4 = Res_Net()

    def forward(self, Q):
        x = torch.rand(Q.size(0), Q.size(1)).cuda()
        grad1 = 2 * torch.matmul(torch.matmul(Q.t(), Q), x) - 2 * Q.t()
        x1 = x - self.alpha1(grad1) + self.beta1(x)

        grad2 = 2 * torch.matmul(torch.matmul(Q.t(), Q), x1) - 2 * Q.t()
        x2 = x1 - self.alpha2(grad2) + self.beta2(x1 - x)

        grad3 = 2 * torch.matmul(torch.matmul(Q.t(), Q), x2) - 2 * Q.t()
        x3 = x2 - self.alpha3(grad3) + self.beta3(x2 - x1)

        grad4 = 2 * torch.matmul(torch.matmul(Q.t(), Q), x3) - 2 * Q.t()
        x4 = x3 - self.alpha4(grad4) + self.beta4(x3 - x2)

        return x4


class UACGFS(nn.Module):
    def __init__(self, d, k, c):
        super(UACGFS, self).__init__()
        self.head = Head(d, k, c)

        self.inv1 = Net_inv()
        self.res_d1 = Net_D(d, k)

        self.miu = nn.Parameter(torch.tensor(1e-4))
        self.ro = nn.Parameter(torch.tensor(2e-4))
        self.eta = nn.Parameter(torch.tensor(1e-3))
        self.nd = nn.Parameter(torch.tensor(1e-8))
        self.beta = nn.Parameter(torch.tensor(1e-8))
        self.gama = nn.Parameter(torch.tensor(1e-8))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, l):
        w, fi, n, I = self.head(x)
        nnt = torch.zeros_like(I).cuda()
        nnt_inv = nnt

        for i in range(3):
            f_x1, f_x_grad1 = init_compute(x, w)
            nnt = torch.matmul(n, n.t())
            nnt_inv = self.inv1(nnt)
            m = torch.matmul(torch.matmul(f_x1, n.t()), nnt_inv)
            n_grad = Grad_N(f_x1, m, n, l, fi, self.gama, self.beta)
            n = n - self.miu * n_grad
            n = torch.relu(n)
            fi_grad = Grad_fi(n)
            fi = fi - self.ro * fi_grad
            # test = self.res_d1(w)
            w_grad = 2 * self.nd * self.res_d1(w) + Grad_W(x, f_x1, f_x_grad1, w, m, n, self.beta)
            w = w - self.eta * w_grad

        loss = (torch.norm(x - torch.matmul(w, torch.tanh(torch.matmul(w.t(),x))), p='fro')**2
                + self.alpha * torch.norm(torch.matmul(nnt, nnt_inv) - I, p='fro') ** 2)

        return w, loss




