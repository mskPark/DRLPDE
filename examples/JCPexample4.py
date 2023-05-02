###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 4: Poiseuille Flow
###

import torch
import math
import numpy as np

############## Global variables ###################

# Pressure Gradient -> Negative to induce downward movement
pressure_constant = -5.0

cylinder_radius = 1.0

############## Collect Errors ######################

collect_error = True
num_error = 2**15
# TODO: Decide num_error automatically based on tolerance

if collect_error:
    def true_fun(X):
        u = torch.stack( ( torch.zeros(X.size(0), device=X.device),
                           torch.zeros(X.size(0), device=X.device),
                           pressure_constant/4/mu*(X[:,0]**2 + X[:,1]**2 - cylinder_radius) ), dim=1)
        return u


# Physical Dimension
x_dim = 3
output_dim = 3

# Steady or Unsteady
t_dim = 0
if t_dim:
    t_range = [[0.0, 1.0]]
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
pde_type = 'NavierStokes'

# Diffusion coefficient
def diffusion(X):
    mu = torch.tensor(1.0 )
    #mu = X[:,4,None]
    return mu

# Forcing term
def forcing(X):
    f = torch.zeros( (X.size(0), output_dim), device=X.device)
    f[:,2] = pressure_constant

    return f

################# Boundary and Initial Conditions ###########
# Use pytorch expressions to make boundary and initial conditions 
#
# To make different boundary conditions for each boundary
#     ensure the correct bdry_con is called when defining the boundaries

def bdry_con(X):
    u = torch.zeros( (X.size(0), output_dim), device=X.device)
    return u

#################  Make the domain  #######################

boundingbox = [ [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0] ]

periodic1 = { 'variable':'z', 
              'base':0.0,
              'top':1.0 }


cylinder1 = {'type':'cylinder',
            'centre': [0.0 ,0.0 ,0.0],
            'radius': 1.0,
            'boundary_condition':bdry_con }
            
list_of_walls = [cylinder1]
list_of_periodic_ends =[periodic1]
solid_walls = []
inlet_outlet = []
mesh = []