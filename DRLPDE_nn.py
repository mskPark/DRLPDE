### Neural Networks

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math

class IncompressibleNN(nn.Module):
    
    ### Incompressible neural network
    ### curl operation built in
    
    def __init__(self, depth, width, x_dim, is_unsteady, **nn_param):
        super(IncompressibleNN, self).__init__()
        
        self.x_dim = x_dim
        self.input_dim = self.x_dim + is_unsteady
        
        dim_out = [1, 3][self.x_dim==3]
        
        modules = []
        modules.append(nn.Linear(self.input_dim, depth))
        for i in range(width - 1):
            modules.append(nn.Linear(depth, depth))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(depth, dim_out))
                       
        self.sequential_model = nn.Sequential(*modules)
    
    def curl(self, a, x):
        if self.x_dim == 2:
            dadx = torch.autograd.grad(a, x, grad_outputs = torch.ones_like(a), create_graph = True, 
                                       retain_graph = True, only_inputs = True)[0]
            u = torch.stack([dadx[:,1], -dadx[:,0]] , dim=1)
            
        elif self.x_dim == 3:
            e = torch.eye(x_dim, device=dev)

            da0dx = torch.autograd.grad(a, x, grad_outputs=e[0,:].repeat(a.size(0), 1), 
                                        create_graph=True, retain_graph = True, only_inputs = True)[0]
            da1dx = torch.autograd.grad(a, x, grad_outputs=e[1,:].repeat(a.size(0), 1),
                                        create_graph=True, retain_graph = True, only_inputs = True)[0]
            da2dx = torch.autograd.grad(a, x, grad_outputs=e[2,:].repeat(a.size(0), 1),
                                        create_graph=True, retain_graph = True, only_inputs = True)[0]

            u = torch.stack([da2dx[:,1] - da1dx[:,2], da0dx[:,2] - da2dx[:,0], da1dx[:,0] - da0dx[:,1] ], dim=1)         
        return u
    
    def forward(self, x):
     
        a = self.sequential_model(x)
        u = self.curl(a, x)
            
        return u


class FeedForwardNN(nn.Module):
    
    ### Feed forward neural network
    
    def __init__(self, depth, width, x_dim, is_unsteady, output_dim, **nn_param):
        super(FeedForwardNN, self).__init__()
        
        self.input_dim = x_dim + is_unsteady
        
        modules = []
        modules.append(nn.Linear(self.input_dim, depth))
        for i in range(width - 1):
            modules.append(nn.Linear(depth, depth))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(depth, output_dim))
        
        self.sequential_model = nn.Sequential(*modules)
        
    def forward(self, x):
        a = self.sequential_model(x)
        
        return a

class ResNetNN(nn.Module):
    
    ### ResNet
    ###
    ### Non-adaptive layers
    
    def __init__(self, depth, width, x_dim, is_unsteady, output_dim, **nn_param):
        super(MyResNet, self).__init__()
        
        input_dim = x_dim + is_unsteady
        
        self.ff_in = nn.Linear(input_dim,depth)
        self.ff_hid = nn.Linear(depth,depth)
        self.ff_out = nn.Linear(depth,output_dim)
        self.activ = nn.Tanh()
        self.numblocks = width
        
    def forward(self, x):
        out = self.activ( self.ff_in(x))
        
        for ii in range(self.numblocks):
            identity = out
            out = self.activ(self.ff_hid(out))
            out = self.ff_hid(out)
            out = out + identity
            out = self.activ(out)

        a = self.ff_out(out)

        return u