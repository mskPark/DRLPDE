###
### This file contains Derivative operators of Neural Networks (as per PINNs)
###

### Sample domain
### Auto-diff at those locations (residual points)
### Create loss term
### Optimization routine
### Learn pressure through gradient potential

### Some problems:
###   - Not generalizable past incompressible flow
###   - Sampling the domain can be improved

import torch
import torch.nn as nn

def Derivative(x, model):

    x.requires_grad_(True)
    u = model(x)

    dev = x.device
    Npoints = x.size(0)

    dim_in = x.size(1)
    dim_out = u.size(1)

    e = torch.eye(dim_out, device=dev)

    Du = torch.empty( (Npoints, dim_out, dim_in), device=dev)

    for ii in range(dim_out):
        Du[:,ii,:] = torch.autograd.grad( u, x, grad_outputs = e[:,ii].repeat(Npoints, 1), create_graph = True, retain_graph = True)[0]
    
    return Du

def Laplacian(x, model, Du):
    x.requires_grad_(True)
    u = model(x)

    dev = x.device
    Npoints = x.size(0)

    dim_in = x.size(1)
    dim_out = u.size(1)

    e = torch.eye(dim_in, device=dev)

    Lap_u = torch.empty( (Npoints, dim_out, dim_in), device=dev)

    for ii in range(dim_out):
        for jj in range(dim_in):
            Lap_u[:,ii,jj] = torch.autograd.grad( Du[:,ii,:], x, grad_outputs = e[jj,:].repeat(Npoints,1), create_graph = True, retain_graph = True)[0][:,jj]
    Lap_u = torch.sum(Lap_u, dim=2)

    return Lap_u