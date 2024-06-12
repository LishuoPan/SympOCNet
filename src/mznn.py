import numpy as np
import torch
import matplotlib.pyplot as plt
import learner as ln
from data import SPData
from nn import SPNN
from PS_method import PSmethod
import os
from time import perf_counter
        
        
class MZNN(SPNN):
    '''NN for solving the optimal control of shortest path in mazes
    '''
    def __init__(self, dim, layers, width, activation, ntype, dr = 0.2, ws = 0, angles = 0.5,
                 phy_dim = 2, qr = 0.1, ql = 1, l = 0.001, eps = 0.1, lam = 1, C = 25, add_dim = 0,
                 ifpenalty = True, rho = 1.0, add_loss = 0, update_lagmul_freq = 10, trajs = 1, 
                 dtype = None, device = None):
        super(MZNN, self).__init__(dim, phy_dim, layers, width, activation, ntype, l, eps, lam, C, add_dim,
        ifpenalty, rho, add_loss, update_lagmul_freq, trajs, dtype, device)
        self.dr = dr                # radius of the drone
        self.ws = ws                # starting points of the obstacles
        self.angles = angles        # angles of the obstacles
        self.num_obs = len(ws)      
        self.qr = qr                # width of the obstacle
        self.ql = ql                # length of the obstacle
    
    # square of distance between q and the line segment connected by w, v, then minus a constant
    def dist_sq(self, w, v, q, ql):
        l2 = ql[None,:,None] ** 2
        q = q.reshape([-1, 1, 2])
        d = torch.sum((q - v[None,...]) * (w - v), dim = -1, keepdims = True) / l2
        # t is the truncation of d into [0,1]
        #t = torch.maximum(torch.zeros_like(d, device = self.device, dtype = self.dtype), torch.minimum(torch.ones_like(d, device = self.device, dtype = self.dtype), d))
        t = 1.0 - torch.relu(1.0- torch.relu(d))
        projection = v[:,None,:] + torch.transpose(t, 0, 1) @ (w - v)[:,None,:]
        d = torch.sum((q.squeeze() - projection) ** 2, dim = -1, keepdims = True)
        d = d.reshape([self.num_obs, -1, self.dim // 2, 1])
        d = torch.transpose(d, 1, 2).reshape([self.num_obs * self.dim // 2, -1, 1]) - (self.qr + self.dr)**2
        return d
    
    # try to avoid for loop in this function
    # return dim: #constraints * (num pts in all trajs) * 1
    def h(self, q):
        ws = torch.tensor(self.ws, device = self.device, dtype = self.dtype)
        ql = torch.tensor(self.ql, device = self.device, dtype = self.dtype)
        angles = torch.tensor(self.angles, device = self.device, dtype = self.dtype)
        vs = ws + torch.stack([torch.cos(angles), torch.sin(angles)], dim = -1) * ql[:,None]
        h = self.dist_sq(ws, vs, q, ql)
        x = q.reshape([-1, self.dim // 2, 2, 1])
        x = torch.transpose(x, 1, 2)
        y = torch.transpose(x, 2, 3)
        z = torch.sum((x - y)**2, dim = 1)
        mask = ~torch.eye(z.shape[1],z.shape[2], dtype=bool)
        min_value = z[:, mask].t()[...,None] - (self.dr * 2) **2
        h = torch.cat([min_value, h], dim = 0)
        return h
    
    def dist_sq_np(self, w, v, q, ql):
        l2 = ql[None,:,None] ** 2
        q = q.reshape([-1, 1, 2])
        d = np.sum((q - v[None,...]) * (w - v), axis = -1, keepdims = True) / l2
        t = np.maximum(np.zeros_like(d), np.minimum(np.ones_like(d), d))
        projection = v[:,None,:] + np.transpose(t, (1, 0, 2)) @ (w - v)[:,None,:]
        d = np.sum((q.squeeze() - projection) ** 2, axis = -1, keepdims = True)
        d = d.reshape([self.num_obs, -1, self.dim // 2, 1])
        d = np.transpose(d, (0,2,1,3)).reshape([self.num_obs * self.dim // 2, -1, 1]) - (self.qr + self.dr)**2
        return d
    
    def h_np(self, q):
        ws = np.array(self.ws)
        ql = np.array(self.ql)
        angles = np.array(self.angles)
        vs = ws + np.stack([np.cos(angles), np.sin(angles)], axis = -1) * ql[:,None]
        h = self.dist_sq_np(ws, vs, q, ql)
        x = q.reshape([-1, self.dim // 2, 2, 1])
        x = np.transpose(x, (0,2,1,3))
        y = np.transpose(x, (0,1,3,2))
        z = np.sum((x - y)**2, axis = 1)
        mask = ~np.eye(z.shape[1],z.shape[2], dtype=bool)
        min_value = (np.transpose(z[:, mask])[...,None]) - (self.dr * 2)**2
        h = np.concatenate([min_value, h], axis = 0)
        return h