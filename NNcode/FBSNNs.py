import numpy as np
from abc import ABC, abstractmethod
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from Models import Sine

from layers import *
from laytest import *

class NeuralNetwork(nn.Module):
    def __init__(self, D, activation_function):
        super().__init__()
        self.activation_function = activation_function

        self.fc1 = StandardLinear(D + 1, 256)
        self.fc2 = StandardLinear(256, 256)
        self.fc3 = StandardLinear(256, 256)
        self.fc4 = StandardLinear(256, 256)
        self.fc5 = StandardLinear(256, 1)

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.activation_function(self.fc3(x))
        x = self.activation_function(self.fc4(x))
        return self.fc5(x)
    

class TensorNetwork(nn.Module):
    def __init__(self, D, activation_function):
        super().__init__()
        self.activation_function = activation_function

        self.fc1 = StandardLinear(D + 1, 8)
        self.mpo1 = MPOLinear3(8, 8, 3, 4)
        self.mpo2 = MPOLinear3(8, 8, 3, 4)
        self.mpo3 = MPOLinear3(8, 8, 3, 4)
        self.fc2 = StandardLinear(8, 1)

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.mpo1(x))
        x = self.activation_function(self.mpo2(x))
        x = self.activation_function(self.mpo3(x))
        return self.fc2(x)

class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, mode, activation):
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        #  We set a random seed to ensure that your results are reproducible
        # torch.manual_seed(0)

        self.Xi = torch.from_numpy(Xi).float().to(self.device)  # initial point
        self.Xi.requires_grad = True

        self.T = T  # terminal time
        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.mode = mode  # architecture: FC, Resnet and NAIS-Net are available
        self.activation = activation
        if activation == "Sine":
            self.activation_function = Sine()
        elif activation == "ReLU":
            self.activation_function = nn.ReLU()

        # initialize NN
        if self.mode == "NN":
            self.model = NeuralNetwork(D, self.activation_function).to(self.device)
        elif self.mode == "TN":
            self.model = TensorNetwork(D, self.activation_function).to(self.device)

        # Record the loss

        self.training_loss = []
        self.iteration = []

    def net_u(self, t, X):  # M x 1, M x D

        input = torch.cat((t, X), 1)
        u = self.model(input)  # M x 1
        Du = torch.autograd.grad(outputs=[u], inputs=[X], grad_outputs=torch.ones_like(u), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]
        return u, Du

    def Dg_tf(self, X):  # M x D

        g = self.g_tf(X)
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]  # M x D
        return Dg

    def loss_function(self, t, W, Xi):
        loss = 0
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)  # M x D
        Y0, Z0 = self.net_u(t0, X0)  # M x 1, M x D

        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
                Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1,
                keepdim=True)
            Y1, Z1 = self.net_u(t1, X1)

            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

            X_list.append(X0)
            Y_list.append(Y0)

        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2))

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        return loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self):  # Generate time + a Brownian motion
        T = self.T

        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N

        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        return t, W

    def train(self, N_Iter, learning_rate):
        f = open("TN8einloss.txt", "a")

        loss_temp = np.array([])

        previous_it = 0
        if self.iteration != []:
            previous_it = self.iteration[-1]

        # Optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        start_time = time.time()
        for it in range(previous_it, previous_it + N_Iter):
            self.optimizer.zero_grad()
            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D

            loss, _, _, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())

            # Save data
            f.write(f"{it} {loss}\n")

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss, Y0_pred, elapsed, learning_rate))
                start_time = time.time()

            # Loss
            if it % 100 == 0:
                self.training_loss.append(loss_temp.mean())
                loss_temp = np.array([])

                self.iteration.append(it)

            graph = np.stack((self.iteration, self.training_loss))

        f.close()
        return graph

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)

        return X_star, Y_star
    def savesmodel(self):
        torch.save(self.model.state_dict(),'model_100_256')
    def displayinfo(self):
        summary(self.model,input_size=(self.D+1,))
    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        pass  # M x1

    @abstractmethod
    def g_tf(self, X):  # M x D
        pass  # M x 1

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)  # M x D x D
    ###########################################################################
