###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 1: Stokes Flow in Disk
###

import torch
import math
import numpy as np

############## Collect Errors ######################

collect_error = True
num_error = 2**15
# TODO: Decide num_error automatically based on tolerance

if collect_error:
    def true_fun(X):
        u = torch.stack( ( -X[:,1], X[:,0] ), dim=1)
        return u

############## Save model and/or Load model ##############

savemodel = 'JCPexample1'
loadmodel = ''

# Physical Dimension
x_dim = 2
output_dim = 2

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
pde_type = 'Stokes'

# Diffusion coefficient
def diffusion(X):
    mu = torch.tensor( 1.0 )
    return mu

# Forcing term
def forcing(X):
    f = torch.zeros( (X.size(0), output_dim), device=X.device)
    return f

################# Boundary and Initial Conditions ###########
# Use pytorch expressions to make boundary and initial conditions 

def bdry_con(X):
    u = torch.stack( ( -X[:,1], X[:,0] ), dim=1)
    return u

#################  Make the domain  #######################

boundingbox = [ [-1.0,1.0], [-1.0,1.0] ]

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
