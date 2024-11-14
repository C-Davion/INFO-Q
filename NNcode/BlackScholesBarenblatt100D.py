import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from torchsummary import summary

from FBSNNs import FBSNN

class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T, M, N, D, mode, activation):
        super().__init__(Xi, T, M, N, D, mode, activation)

    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    def g_tf(self, X):  # M x D
        return torch.sum(X ** 2, 1, keepdim=True)  # M x 1

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        return 0.4 * torch.diag_embed(X)  # M x D x D

    ###########################################################################

def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    r = 0.05
    sigma_max = 0.4
    return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1

def run_model(model, N_Iter, learning_rate,D):
    tot = time.time()
    samples = 5
    print(model.device)
    #print("number of parameters:",summary(model,(D+1,)))
    graph = model.train(N_Iter, learning_rate)
    print("total time:", time.time() - tot, "s")
    ### Save the model
    #model.savesmodel()
    ### Display info
    #model.displayinfo()
    ###
    np.random.seed(42)
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()

    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, -1, 1])

    file_name=f'tn8eindatanew{D}_{learning_rate}_{N_Iter}.npz'
    np.savez(file_name, X_pred=X_pred, Y_pred= Y_pred, Y_test=Y_test,t_test=t_test)
    print("dump complete into",file_name)
if __name__ == "__main__":
    tot = time.time()
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions

    layers = [D + 1] + 4 * [256] + [1]

    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
    T = 1.0

    "Available architectures"
    activation = "Sine"  # sine and ReLU are available
    NN_model = BlackScholesBarenblatt(Xi, T,
                                   M, N, D,
                                   "NN", activation)
    TN_model = BlackScholesBarenblatt(Xi, T,
                                   M, N, D,
                                   "TN", activation)
    #run_model(NN_model, 2*10**4, 1e-3,D)
 
    run_model(TN_model, 2*10**4, 1e-3,D)
    print("end")