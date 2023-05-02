###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 7: Taylor Green Vortex

### u(x,y,t) = cos(x) sin(y) e^{-2 \nu t}
### v(x,y,t) = -sin(x) cos(y) e^{-2 \nu t}
### p(x,y,t) = - \rho/4 (cos(2x) + cos(2y)) e^{-4\nu t}

import torch
import math
import numpy as np

mu = 1.0

############## Collect Errors ######################

collect_error = True
num_error = 2**15
# TODO: Decide num_error automatically based on tolerance

if collect_error:
    def true_fun(X):
        u = torch.stack( ( torch.cos(X[:,0])*torch.sin(X[:,1])*torch.exp(-2*mu*X[:,2]),
                          -torch.sin(X[:,0])*torch.cos(X[:,1])*torch.exp(-2*mu*X[:,2]) ), dim=1)
        return u

############## Problem Parameters ################


# Physical Dimension
x_dim = 2
output_dim = 2

# Steady or Unsteady
t_dim = 1
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

pde_type = 'viscousBurgers'

# Diffusion coefficient
def diffusion(X):
    mu = torch.tensor( 1.0 )
    #mu = X[:,4,None]
    return mu

# Forcing term
def forcing(X):
    f = torch.zeros( (X.size(0), output_dim), device=X.device)
    return f

################# Boundary and Initial Conditions ###########
# Use pytorch expressions to make boundary and initial conditions 
#
# To make different boundary conditions for each boundary
#     ensure the correct bdry_con is called when defining the boundaries

def bdry_con(X):
    u = torch.zeros( (X.size(0), output_dim), device=X.device)
    return u

def init_con(X):
    u = torch.stack( ( torch.cos(X[:,0])*torch.sin(X[:,1]),
                       -torch.sin(X[:,0])*torch.cos(X[:,1]) ), dim=1)
    return u

#################  Make the domain  #######################

boundingbox = [ [-math.pi, math.pi], [-math.pi,math.pi], ]

periodic1 = { 'variable':'x', 
              'base':-math.pi,
              'top':math.pi }

periodic2 = { 'variable':'y', 
              'base':-math.pi,
              'top':math.pi }

list_of_walls = []
list_of_periodic_ends =[periodic1, periodic2]
solid_walls = []
inlet_outlet = []
mesh = []