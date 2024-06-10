#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:18:56 2021

@author: zen
"""
import learner as ln
import numpy as np
import torch
from learner.utils import mse, grad
from time import perf_counter
import torch.nn.functional as F
import torch.nn as nn

class ParametricNN(ln.nn.Module):
    def __init__(self, latent_dim, layers, width, activation):
        super(ParametricNN, self).__init__()
        self.latent_dim = latent_dim # 8
        # self.__init_net(layers, width, activation)
        self.fc1 = torch.nn.Linear(self.latent_dim, 128, bias=True)
        self.fc2 = torch.nn.Linear(128, 128, bias=True)
        self.fc3 = torch.nn.Linear(128, 128, bias=True)
        self.fc4 = torch.nn.Linear(128, self.latent_dim, bias=True)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        self.__init_weights()

    def forward(self, x):
        x = x.view(-1, self.latent_dim)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        x = x.view(-1, 1, self.latent_dim)
        return x

    # def __init_net(self, layers, width, activation):
    #     net = torch.nn.Sequential(
    #         torch.nn.Linear(self.latent_dim, 2 * self.latent_dim),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(2 * self.latent_dim, 2 * self.latent_dim),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(2 * self.latent_dim, 2 * self.latent_dim),
    #         torch.nn.ReLU(),
    #         # torch.nn.Linear(128, 128),
    #         # torch.nn.ReLU(),
    #         torch.nn.Linear(2 * self.latent_dim, self.latent_dim),
    #     )
    #
    #     self.net = net

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform(m.weight)
            # if isinstance(m, torch.nn.BatchNorm1d):
            #     nn.init.xavier_uniform(m.weight)
        
class ParametersNN(ln.nn.Module):
    def __init__(self, latent_dim, layers, width, activation):
        super(ParametersNN, self).__init__()
        self.Qslope = ParametricNN(latent_dim, layers, width, activation)
        self.Qincpt = ParametricNN(latent_dim, layers, width, activation)
        self.Pincpt = ParametricNN(latent_dim, layers, width, activation)

    def forward(self, x):
        Qslope = self.Qslope(x)
        Qincpt = self.Qincpt(x)
        Pincpt = self.Pincpt(x)
        return {'Qslope': Qslope, 'Qincpt': Qincpt, 'Pincpt': Pincpt}
    
    # def criterion(self, X, y):
    #     Q = self.Qslope(X['interval']) + self.Qincpt(X['interval'])
    #     P = 0.0 * X['interval'] + self.Pincpt(X['interval'])
    #     loss_Q = ((Q - y['Q_target']) ** 2).mean()
    #     loss_P = ((P - y['P_target']) ** 2).mean()
    #     loss = loss_Q + loss_P
    #     return loss

class QslopeNet(ln.nn.Module):
    def __init__(self, latent_dim, layers, width, activation):
        super(QslopeNet, self).__init__()
        self.Qslope = ParametricNN(latent_dim, layers, width, activation)

    def forward(self, x):
        Qslope = self.Qslope(x)
        return Qslope

class QincptNet(ln.nn.Module):
    def __init__(self, latent_dim, layers, width, activation):
        super(QincptNet, self).__init__()
        self.Qincpt = ParametricNN(latent_dim, layers, width, activation)

    def forward(self, x):
        Qincpt = self.Qincpt(x)
        return Qincpt

class PincptNet(ln.nn.Module):
    def __init__(self, latent_dim, layers, width, activation):
        super(PincptNet, self).__init__()
        self.Pincpt = ParametricNN(latent_dim, layers, width, activation)

    def forward(self, x):
        Pincpt = self.Pincpt(x)
        return Pincpt

class SPNN(ln.nn.LossNN):
    '''NN for solving the optimal control of shortest path with obstacles
    '''
    def __init__(self, dim, phy_dim, layers, width, activation, ntype, l, eps, lam, C, add_dim, ifpenalty, rho, add_loss, update_lagmul_freq, trajs, dtype, device):
        super(SPNN, self).__init__()
        self.dim = dim
        self.phy_dim = phy_dim
        self.ntype = ntype          # (LA/G)SympNet or FNN
        self.dtype = dtype
        self.device = device
        self.l = l                  # hyperparameter controling the soft penalty
        self.eps = eps              # hyperparameter controling the soft penalty
        self.lam = lam              # weight of the BC
        self.C = C                  # speed limit
        self.add_dim = add_dim      # added dimension
        self.ifpenalty = ifpenalty  # True for using penalty, False for augmented Lagrangian
        self.latent_dim = add_dim + dim
        self.add_loss = add_loss    # 0 for no added loss, 1 for aug lag / log penalty, 2 for quad penalty
        # parameters for Lag mul begins
        # Lagrange multiplier for h in opt ctrl prob. Will be a vector in later update
        self.lag_mul_h = torch.zeros(1,dtype=self.dtype, device=self.device)
        # Lagrange multiplier for boundary condition in training process. NOTE: assume two pts bc
        self.lag_mul_bc = torch.zeros(trajs,2,self.dim,dtype=self.dtype, device=self.device) 
        self.rho_h = rho	    # parameter for augmented Lagrangian for h
        self.rho_bc = rho	    # parameter for augmented Lagrangian for bc
        self.update_lagmul_freq = update_lagmul_freq
        self.update_lagmul_count = 0
        self.eta0 = 0.1		    # initial tol for aug lag
        self.etak_h = self.eta0	    # k-th tol for aug lag for h
        self.etak_bc = self.eta0    # k-th tol for aug lag for bc
        # parameters for Lag mul ends

        self.trajs = trajs
        # self.__init_param()
        self.__init_net(layers, width, activation, ntype)
        
    # X['interval'] is num * 1
    def criterion(self, X, y):
        # self.params['Qslope'] is trajs * 1 * latent_dim
        # self.params['Qincpt'] is trajs * 1 * latent_dim
        # self.params['Pincpt'] is trajs * 1 * latent_dim
        
        # self.params = self.parameters_nn(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Qslope = self.qslope_net(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Qincpt = self.qincpt_net(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Pincpt = self.pincpt_net(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Q = Qslope * X['interval'] + Qincpt
        P = 0.0 * X['interval'] + Pincpt
        QP = torch.cat([Q,P], axis = -1).reshape([-1, self.latent_dim * 2])
        qp = self.net(QP)
        H = self.H(qp)  # (trajs*num) *1
        dH = grad(H, qp)  # (trajs*num) * (2latent_dim)
        grad_output = Qslope.repeat([1,QP.shape[0]//self.trajs, 1]).reshape([-1, self.latent_dim])
        grad_output1 = torch.cat([grad_output,torch.zeros_like(grad_output)], dim = -1)
        jacob = torch.autograd.functional.jvp(self.net, QP, grad_output1, create_graph=True)[1]
        loss_1 = mse(jacob[:, :self.latent_dim], dH[...,self.latent_dim:])
        loss_2 = mse(jacob[:, self.latent_dim:], -dH[...,:self.latent_dim])
        loss_sympnet = loss_1 + loss_2
        # print("loss_sympnet: ", loss_sympnet)

        loss_bd = self.bd_loss(X, y)
        loss = loss_sympnet + self.lam * loss_bd
        # print("loss_bd", loss_bd)

        # aug Lag: ||max(0, mul - rho * h(q))||^2/ (2*rho)
        loss_aug_lag = torch.sum(torch.relu(self.lag_mul_h - self.rho_h * self.h(qp[...,:self.dim]))**2)/(2*self.rho_h) # augmented Lagrangian
        loss = loss + loss_aug_lag
        # print("aug Lag: ", loss_aug_lag)

        # loss for bd
        y_m_bdq = y['bd'] - self.predict_q(X['bd'])
        loss_aug_bd = torch.nn.MSELoss(reduction='sum')(self.lag_mul_bc, self.rho_bc * y_m_bdq)/(2*self.rho_bc)
        loss = loss + loss_aug_bd
        # print("loss_aug_bd: ", loss_aug_bd)
        return loss
    
    # MSE loss of bdry condition
    def bd_loss(self, X, y):
        bdq = self.predict_q(X['bd'])
        return mse(bdq, y['bd'])
    
    # MSE of bd err + sum of |min(h(q),0)|^2 (i.e., penalty method using quadratic)
    # def con_loss(self, X, y):
    #     Q = self.params['Qslope'] * X['interval'] + self.params['Qincpt']
    #     P = 0.0 * X['interval'] + self.params['Pincpt']
    #     QP = torch.cat([Q,P], axis = -1).reshape([-1, self.latent_dim * 2])
    #     q = self.net(QP)[...,:self.dim]
    #     con_loss = torch.mean(torch.relu(-self.h(q))**2)
    #     return self.bd_loss(X,y) + con_loss
    
    # prediction without added dims
    def predict(self, t, returnnp=False):
        # TODO: this is not working, need to change self.params
        Q = self.params['Qslope'] * t + self.params['Qincpt']
        P = 0.0 * t + self.params['Pincpt']
        QP = torch.cat([Q,P], dim = -1)
        qp = self.net(QP)
        q = qp[...,:self.dim]
        p = qp[...,self.latent_dim:self.latent_dim+self.dim]
        qp = torch.cat([q,p], dim = -1)
        if returnnp:
            qp = qp.detach().cpu().numpy()
        return qp
    
    # prediction q without added dims
    def predict_q(self, t, returnnp=False):
        Qslope = self.qslope_net(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Qincpt = self.qincpt_net(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Pincpt = self.pincpt_net(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Q = Qslope * t + Qincpt
        P = 0.0 * t + Pincpt
        QP = torch.cat([Q,P], dim = -1)
        qp = self.net(QP)
        q = qp[...,:self.dim]
        if returnnp:
            q = q.detach().cpu().numpy()
        return q
        
    # t is num * 1
    def predict_v(self, t, returnnp=False):
        Qslope = self.qslope_net(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Qincpt = self.qincpt_net(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Pincpt = self.pincpt_net(torch.ones((self.trajs, 1, self.latent_dim), dtype=self.dtype, device=self.device))
        Q = Qslope * t + Qincpt
        P = 0.0 * t + Pincpt
        QP = torch.cat([Q, P], axis=-1).reshape([-1, self.latent_dim * 2])
        qp = self.net(QP)    
        grad_output = Qslope.repeat([1,QP.shape[0]//self.trajs, 1]).reshape([-1, self.latent_dim])
        grad_output1 = torch.cat([grad_output,torch.zeros_like(grad_output)], dim = -1)
        v = torch.autograd.functional.jvp(self.net, QP, grad_output1, create_graph=True)[1][:,:self.latent_dim].unsqueeze(0)
        if returnnp:
            v = v.detach().cpu().numpy()
        return v
    
    def LBFGS_training(self, X, y, returnnp=False, lbfgs_step = 0):
        from torch.optim import LBFGS, Adam
        start = perf_counter()
        optim_bd = LBFGS([param for param in self.parameters_nn.Qslope.parameters()] +
                         [param for param in self.parameters_nn.Qincpt.parameters()] +
                         [param for param in self.parameters_nn.Pincpt.parameters()], history_size=100,
                        max_iter=10,
                        tolerance_grad=1e-08, tolerance_change=1e-09,
                        line_search_fn="strong_wolfe")
        optim = optim_bd
        # change self.penalty to True s.t. there is no aug Lag in loss
        self.penalty = True
        loss_fnc = self.criterion  # use the same loss as in previous nn training
        for i in range(lbfgs_step):
            def closure():
                if torch.is_grad_enabled():
                    optim.zero_grad()
                loss = loss_fnc(X, y)
                if i % 10 == 0:
                    print('{:<9} loss: {:<25}'.format(i, loss.item()), flush=True)
                if loss.requires_grad:
                    loss.backward()
                return loss
            optim.step(closure)
        end = perf_counter()
        execution_time = (end - start)
        print('LBFGS running time: {}'.format(execution_time), flush=True)
    
    # penalty function: if x>l, return -log(x); else return -log(l)+1/2*(((x-2l)/l)^2-1)
    def betal(self, x):
        return torch.max(torch.where(x > self.l, -torch.log(torch.clamp(x, self.l/2)), - np.log(self.l) + 0.5 * (((x - 2*self.l) / self.l) ** 2 - 1)), dim=0)[0]

    # if qp is (trajs*num) * (2latent_dim), then H is (trajs*num) * 1
    def H(self, qp):
        q = qp[...,:self.dim]
        p = qp[...,self.latent_dim:self.latent_dim + self.dim]
        p_dummy = qp[...,self.latent_dim + self.dim:]
        p2 = torch.sum(p.reshape([-1, self.dim // self.phy_dim, self.phy_dim]) ** 2, dim = -1)
        # H1 is for real dimensions: sum over all drones, if |p|<C, return |p|^2/2; else return C|p| - C^2/2
        H1 = torch.sum(torch.where(p2 < self.C ** 2, p2/2, self.C*torch.sqrt(p2) - self.C**2/2), dim = -1, keepdims = True)
        #H1 = torch.sum(p2/2, dim = -1, keepdims = True)
        # H2 is negative of the added cost (log penalty of h)
        #H2 = - self.eps * self.betal(self.h(q))  # eps * beta_l(h(q))
        # H3 is for dummy variables: |p|^2/2
        H2 = 0
        H3 = torch.sum(p_dummy ** 2, dim = -1, keepdims = True) / 2
        return H1 + H2 + H3

    def update_lag_mul(self, t, bdt, bdy):
        self.update_lagmul_count = self.update_lagmul_count + 1
        # update Lag mul after update_lagmul_freq * print_every steps of training
        if self.ifpenalty == False and self.update_lagmul_count % self.update_lagmul_freq == 0:
            eta_star = 0.001
            alp, beta = 0.5, 0.5
            tau = 1.1
            # compute constraint h
            q = self.predict_q(t)
            h = self.h(q)
            # compute constraint bc
            bdq = self.predict_q(bdt)
            # update lag_mul for h and bc
            lag_mul_h, lag_mul_bc = self.lag_mul_h, self.lag_mul_bc
            # mul <- max(mul - rho*h, 0)
            new_lag_mul_h = torch.relu(lag_mul_h - self.rho_h * h).detach()
            # mul <- mul + rho*(bdq-y)
            new_lag_mul_bc = (lag_mul_bc + self.rho_bc*(bdq - bdy)).detach()
            # hard constraint: contraint_val == 0
            constraint_h = (new_lag_mul_h - lag_mul_h) / self.rho_h
            constraint_bc = bdq - bdy

            def update_lag_mul_framework(constraint_val, etak, lag_mul, new_lag_mul, rho):
                ret_lag_mul = lag_mul
                ret_etak = etak
                ret_rho = rho
                if torch.max(torch.abs(constraint_val)) < max(eta_star, etak):
                    # update lag mul
                    ret_lag_mul = new_lag_mul
                    ret_etak = etak / (1 + rho ** beta)
                    print('update lag mul step {}, etak {}'.format(torch.max(torch.abs(ret_lag_mul - lag_mul)).item(), ret_etak))
                else:
                    ret_rho = rho * tau
                    ret_etak = self.eta0 / (1+ rho ** alp)
                    print('update rho {}, etak {}'.format(ret_rho, ret_etak))
                return ret_lag_mul, ret_etak, ret_rho

            self.lag_mul_h, self.etak_h, self.rho_h = update_lag_mul_framework(constraint_h, self.etak_h, lag_mul_h, new_lag_mul_h, self.rho_h)
            self.lag_mul_bc, self.etak_bc, self.rho_bc = update_lag_mul_framework(constraint_bc, self.etak_bc, lag_mul_bc, new_lag_mul_bc, self.rho_bc)
    
    # v is ... * dim, L is ... * 1
    def L(self, v): # running cost: sum of |v|^2/2
        return torch.sum(v**2/2, dim=-1, keepdim = True)
    
    def hmin_function(self, t, traj_count): # compute the min value of constraint function h among the first traj_count many trajs
        q = self.predict_q(t)
        h = self.h(q)
        hmin,_ = torch.min(h, dim=0)
        hmin = hmin.reshape([self.trajs, -1])
        hmin = torch.min(hmin[:traj_count, :])
        return hmin

    # t is num * 1 and assume t is grid points
    # return size (trajs)
    def value_function(self, t): # compute the value function (ignore constraints)
        dt = (t[-1,0] - t[0,0]) / (list(t.size())[-2] - 1)  # a scalar
        v = self.predict_v(t)   # trajs * num * dim
        L = self.L(v)           # trajs * num * 1
        L[:,0,:] = L[:,0,:]/2
        L[:,-1,:] = L[:,-1,:]/2
        cost = torch.sum(L[...,0], -1) * dt
        return cost
        
    def __init_param(self):
        params = torch.nn.ParameterDict()
        params['Qincpt'] = torch.nn.Parameter(torch.ones((self.trajs, 1, self.latent_dim)))
        params['Qslope'] = torch.nn.Parameter(torch.ones((self.trajs, 1, self.latent_dim)))
        params['Pincpt'] = torch.nn.Parameter(torch.ones((self.trajs, 1, self.latent_dim)))
        self.params = params
        
    def __init_net(self, layers, width, activation, ntype):
        if ntype == 'G':
           self.net = ln.nn.GSympNet(self.latent_dim*2, layers, width, activation)
        elif ntype == 'LA':
           self.net = ln.nn.LASympNet(self.latent_dim*2, layers, width, activation)
        elif ntype == 'FNN':
           self.net = ln.nn.FNN(self.latent_dim*2, self.latent_dim*2, layers, width, activation)
           
        self.parameters_nn = ParametersNN(self.latent_dim, layers, width, activation)
        self.qslope_net = QslopeNet(self.latent_dim, layers, width, activation)
        self.qincpt_net = QincptNet(self.latent_dim, layers, width, activation)
        self.pincpt_net = PincptNet(self.latent_dim, layers, width, activation)