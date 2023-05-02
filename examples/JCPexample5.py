###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 5: Cavity Flow
###

import torch
import math
import numpy as np

# Physical Dimension
x_dim = 2
output_dim = 2

lidspeed = 1.0

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
    mu = torch.tensor(1.0 )
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

def bdry_con_wall(X):
    u = torch.zeros( (X.size(0), output_dim), device=X.device)
    return u

def bdry_con_lid(X):
    u = torch.zeros( (X.size(0), output_dim), device=X.device)

    u[:,0] = lidspeed
    return u

#################  Make the domain  #######################

boundingbox = [ [-1.0,1], [-1,1] ]

wall_left = {'type':'line',
             'point': [-1,0],
             'normal': [1,0],
             'endpoints': [ [-1,-1], [-1,1] ],
             'boundary_condition': bdry_con_wall }

lid_top = { 'type':'line',
             'point': [0,1],
             'normal': [0,-1],
             'endpoints': [ [-1,1],   [1,1]  ],
             'boundary_condition': bdry_con_lid }

wall_right= {'type':'line',
             'point': [1, 0],
             'normal':  [-1,0],
             'endpoints':  [ [1,-1],   [1,1]  ],
             'boundary_condition': bdry_con_wall }

wall_bot = {'type':'line',
             'point': [0,-1],
             'normal': [0,1],
             'endpoints': [ [-1,-1], [1,-1] ],
             'boundary_condition': bdry_con_wall }

list_of_walls = [wall_left, wall_right, wall_bot, lid_top]
list_of_periodic_ends =[]
solid_walls = []
inlet_outlet = []
mesh = []