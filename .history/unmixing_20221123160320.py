import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cvxpy as cp
class L1NMF_Net(nn.Module):
    def __init__(self, layerNum, M, A):
        super(L1NMF_Net, self).__init__()
        R = np.size(M, 1)
        eig, _ = np.linalg.eig(M.T @ M)
        eig += 0.1
        L = 1 / np.max(eig)
        theta = np.ones((1, R)) * 0.01 * L
        # Endmember
        eig, _ = np.linalg.eig(A @ A.T)
        eig += 0.1
        L2 = np.max(eig)
        L2 = 1 / L2

        self.p = nn.ParameterList()
        self.L = nn.ParameterList()
        self.theta = nn.ParameterList()
        self.L2 = nn.ParameterList()
        self.W_a = nn.ParameterList()
        self.layerNum = layerNum
        temp = self.calW(M)
        for k in range(self.layerNum):
            self.L.append(nn.Parameter(torch.FloatTensor([L])))
            self.L2.append(nn.Parameter(torch.FloatTensor([L2])))
            self.theta.append(nn.Parameter(torch.FloatTensor(theta)))
            self.p.append(nn.Parameter(torch.FloatTensor([0.5])))
            self.W_a.append(nn.Parameter(torch.FloatTensor(temp)))
        self.layerNum = layerNum
    def forward(self, X, _M, _A):
        self.W_m = torch.FloatTensor(_A)
        M = list()
        M.append(torch.FloatTensor(_M))
        A = list()
        A.append(torch.FloatTensor(_A.T))
        for k in range(self.layerNum):
            theta = self.theta[k].repeat(A[-1].size(1), 1).T
            T = M[-1].mm(A[-1]) - X
            _A = A[-1] - self.L[k]*self.W_a[k].T.mm(T)
            _A = self.sum2one(F.relu(self.self_active(_A, self.p[k], theta)))
            A.append(_A)
            T = M[-1].mm(A[-1]) - X
            _M = M[-1] - T.mm(self.L2[k] * self.W_m)
            _M = F.relu(_M)
            M.append(_M)
        return M, A

    def self_active(self, x, p, lam):
        tau=pow(2*(1-p)*lam,1/(2-p))+p*lam*pow(2*lam*(1-p), (p-1)/(2-p))
        v = x
        ind = (x-tau) > 0
        ind2=(x-tau)<=0
        v[ind]=x[ind].sign() * (x[ind].abs() - p * lam[ind] * pow(x[ind].abs(), p - 1))
        v[ind2]=0
        v[v>1]=1
        return v
    def calW(self,D):
        (m,n)=D.shape
        W = cp.Variable(shape=(m, n))
        obj = cp.Minimize(cp.norm(W.T @ D, 'fro'))
        # Create two constraints.
        constraint = [cp.diag(W.T @ D) == 1]
        prob = cp.Problem(obj, constraint)
        result = prob.solve(solver=cp.SCS, max_iters=1000)
        print('residual norm {}'.format(prob.value))
        # print(W.value)
        return W.value
    def sum2one(self, Z):
        temp = Z.sum(0)
        temp = temp.repeat(Z.size(0), 1) + 0.0001
        return Z / temp
if __name__ == '__main__':  
    A0 = np.random.randn(224, 6)
    S0 = np.random.randn(6, 4096)
    model = L1NMF_Net(9, A0, S0)
    checkpoint = torch.load("./net_params_1123500.pkl")
    dataFile = ".//Data//syntheticDataNewSNR20dB20170601.mat"
    data = scio.loadmat(dataFile)
    X = data['x_n']
    A = data['A']
    s = data['s']
    X = torch.tensor(X, dtype = torch.float32)
    # A = torch.tensor(A)  
    s = torch.tensor(s, dtype = torch.float32)
    model.load_state_dict(checkpoint)
    model.eval()
    # O1, O2 = checkpoint(X, A, s)
    O3, O4 = model(X, A, s) 
    print("f")
