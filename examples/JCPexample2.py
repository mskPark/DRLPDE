###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 2: Unsteady Stokes Flow in Disk
###

import torch
import math
import numpy as np

import scipy.special as bessel

v0 = 1.0

############## Collect Errors ######################

collect_error = True
num_error = 2**15

if collect_error:
    def true_fun(X):
        Truncate = 20
        roots = bessel.jn_zeros(1,Truncate)
        
        c = np.zeros(Truncate)
        
        u = torch.stack( ( -X[:,1], X[:,0] ), dim=1)
        
        r = torch.norm(X, dim=1)
        th = np.angle( np.complex( X[:,0].detach().cpu().numpy(), X[:,1].detach().cpu().numpy() ) )
        
        for jj in range(Truncate):
            c[jj] = 2*(-bessel.j0(roots[jj])/roots[jj])/(bessel.jv(2,roots[jj])**2)
            u[:,0] += v0*c[jj]*torch.sin(th)*torch.tensor(bessel.j1(roots[jj]*r.detach().cpu().numpy()), device=X.device)*torch.exp(-mu*roots[jj]**2*X[:,2])
            u[:,1] += -v0*c[jj]*torch.cos(th)*torch.tensor(bessel.j1(roots[jj]*r.detach().cpu().numpy()), device=X.device)*torch.exp(-mu*roots[jj]**2*X[:,2])
                    
        return u

# Physical Dimension
x_dim = 2
output_dim = 2

# Steady or Unsteady
t_dim = 0
if t_dim:
    t_range = [[0.0, 0.25]]
else:
    t_range = [ [] ]

# Hyperparameters
hyper_dim = 0
if hyper_dim:
    hyper_range = [[0.0, 1.0], [1.0, 5.0]]
else:
    hyper_range = [ [] ]

################# PDE Coefficients ########################

# PDE type:
pde_type = 'Stokes'

# Diffusion coefficient
mu = 1

# Forcing term
def forcing(X):
    f = torch.zeros( (X.size(0), output_dim), device=X.device)
    return f

################# Boundary and Initial Conditions ###########
# Use pytorch expressions to make boundary and initial conditions 

def bdry_con(X):
    u = torch.stack( ( -X[:,1], X[:,0] ), dim=1)
    return u

def init_con(X):
    u = torch.zeros( ( X.size(0),output_dim), device=X.device)
    return u


#################  Make the domain  #######################

boundingbox = [ [-1.0, 1.0], [-1.0, 1.0] ]

ring1 = {   'type':'ring',
            'centre': [0.0 ,0.0],
            'radius': 1.0,
            'endpoints': [],
            'boundary_condition':bdry_con }
            
list_of_walls = [ring1]
list_of_periodic_ends =[]
solid_walls = []
inlet_outlet = []
mesh = []