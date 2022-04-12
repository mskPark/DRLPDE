###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 5: Cavity Flow
###

import torch
import math
import numpy as np

############## Save model and/or Load model ##############

savemodel = 'JCPexample5'
loadmodel = ''

# Physical Dimension
x_dim = 2
output_dim = 2

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

    u[:,0] = 1.0
    return u

#################  Make the domain  #######################
#     First define a bounding box containing your domain
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


boundingbox = [ [-1,1], [-1,1] ]

wall_left   = [ 'line', [-1,-1], [1,0], [ [-1,-1], [-1,1] ], bdry_con_wall ]
lid_top     = [ 'line', [-1,1],  [0,-1],  [ [-1,1],   [1,1]  ], bdry_con_lid ]
wall_right  = [ 'line', [1, -1], [-1,0],  [ [1,-1],   [1,1]  ], bdry_con_wall ]
wall_bot    = [ 'line', [-1, -1], [0,1], [ [-1,-1], [1,-1]  ], bdry_con_wall ]

my_bdry = [wall_left, lid_top, wall_right, wall_bot ]

is_periodic = False