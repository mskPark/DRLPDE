###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 5: Cavity Flow
###

import torch
import math
import numpy as np

global lidspeed

############## Save model and/or Load model ##############

savemodel = 'JCPexample5'
loadmodel = ''

# Physical Dimension
x_dim = 2
output_dim = 2

lidspeed = 1.0

# Steady   or Unsteady
# Elliptic or Parabolic
is_unsteady = False
input_dim = x_dim + is_unsteady

# Is there a true solution
exists_analytic_sol = False

# Provide contour levels
plot_levels = np.linspace(-1,1,100)

def true_solution(X):
    pass


################# PDE Coefficients ########################

# PDE type:
pde_type = 'StokesFlow'

# Diffusion coefficient
mu = 1

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

boundingbox = [ [-1,1], [-1,1] ]

wall_left = {'type':'line',
             'point': [-1,-1],
             'normal': [1,0],
             'endpoints': [ [-1,-1], [-1,1] ],
             'boundary_condition': bdry_con_wall }

lid_top = { 'type':'line',
             'point': [-1,1],
             'normal': [0,-1],
             'endpoints': [ [-1,1],   [1,1]  ],
             'boundary_condition': bdry_con_lid }

wall_right= {'type':'line',
             'point': [1, -1],
             'normal':  [-1,0],
             'endpoints':  [ [1,-1],   [1,1]  ],
             'boundary_condition': bdry_con_wall }

wall_bot = {'type':'line',
             'point': [-1,-1],
             'normal': [0,1],
             'endpoints': [ [-1,-1], [1,-1] ],
             'boundary_condition': bdry_con_wall }

list_of_dirichlet_boundaries = [wall_left, lid_top, wall_right, wall_bot ]
list_of_periodic_boundaries =[]