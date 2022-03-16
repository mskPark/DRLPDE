###
### Parameters
###

import torch
import math
import numpy as np

use_cuda = torch.cuda.is_available()
dev = torch.device("cuda:0" if use_cuda else "cpu")

# Physical Dimension
x_dim = 2
output_dim = 2

# Steady   or Unsteady
# Elliptic or Parabolic
is_unsteady = False
input_dim = x_dim + is_unsteady

# Give the time range
time_range = [0.0, 1.0]
    
################# PDE Coefficients ########################

# PDE type:
#     NavierStokes, (TODO: Elliptic, Parabolic)
pde_type = 'NavierStokes'

# Diffusion coefficient
mu = 0.1

# Forcing term
def forcing(X):
    f = torch.zeros( (X.size(0), output_dim), device=X.device)
    return f

# Drift coefficient for Elliptic/Parabolic PDES
def drift(X):
    drift = torch.zeros( (X.size(0), output_dim), device=X.device)
    return drift

# Reaction coefficient for Elliptic/Parabolic PDES
def reaction(X):
    reaction = torch.zeros( (X.size(0), output_dim), device=X.device)
    return reaction

################# Boundary and Initial Conditions ###########
# Use pytorch expressions to make boundary and initial conditions 
#
# To make different boundary conditions for each boundary
#     ensure the correct bdry_con is called when defining the boundaries

def bdry_con(X):
    return torch.zeros( (X.size(0), output_dim), device=X.device )

def init_con(X):
    return torch.zeros( (X.size(0), output_dim), device=X.device )


#################  Make the domain  #######################
#     First define a large rectangular region containing your domain
#         Syntax: [ x interval, y interval, z interval ]
#         Points will be sampled through rejection sampling
#
#     Define each boundary
#         lines: [ 'line', point, normal, endpoints, bdry_condition ]
#         disk:  [ 'disk', centre, radius, endpoints, bdry_condition]
#         
#     Intersections of boundaries must be input manually
#         These should be ordered as points will be sampled from first to second
#         only 2 intersection points allowed
#         
#     Boundary condition is given by a function using pytorch expressions


domain = [ [-3,3], [-2,2] ]

centre1 = [-1,0]
radius1 = 0.75
endpoints1 = []

bdry1 = [ 'disk', centre1, radius1, endpoints1, bdry_con ]

point2 = [-3,1]
normal2 = [-1,3]
endpoints2 = [ [-3,1], [0,2] ]

bdry2 = [ 'line', point2, normal2, endpoints2, bdry_con ]

inlet_left   = [ 'line', [-3,-2], [-1,0], [ [-3,-2], [-3,1] ], bdry_con ]
wall_top     = [ 'line', [-3,2],  [0,1],  [ [0,2],   [3,2]  ], bdry_con ]
outlet_right = [ 'line', [3, -2], [1,0],  [ [3,0],   [3,2]  ], bdry_con ]
wall_bot     = [ 'line', [0, -1], [1,-3], [ [-3,-2], [3,0]  ], bdry_con ]

my_bdry = [bdry1, bdry2, inlet_left, wall_top, outlet_right, wall_bot ]

############## Save model and/or Load model ##############

savemodel = 'Test'
loadmodel = ''

############## Walker and Boundary Parameters ############

# Time step
dt = 1e-4

# exit tolerance
tol = 1e-6

# Number of walkers
num_walkers = 2**12
num_ghost = 128
num_batch = 2**6

# Update walkers
# Options: 
#    move -- moves walkers to one of their new locations
#    remake -- remake walkers at each training step
#    fixed -- keeps walkers fixed
update_walkers = 'remake'
update_walkers_every = 1

# Number of boundary points 
num_bdry = 2**8
num_batch_bdry = 2**6

############## Training Parameters #######################

# Training epochs
num_epoch = 10
update_print_every = 1

# Neural Network Architecture
nn_depth = 64
nn_width = 4

# Weighting of losses
lambda_bell = 1e-2/dt
lambda_bdry = 1e2
lambda_init = 1e2

# Learning rate
learning_rate = 1e-2
adam_beta = (0.9,0.999)
weight_decay = 1e-4

