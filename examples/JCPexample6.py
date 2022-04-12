###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 6: Steady Flow Past Disk
###

import torch
import math
import numpy as np

global L_height, v0

############## Save model and/or Load model ##############

savemodel = 'JCPexample6'
loadmodel = ''

# Physical Dimension
x_dim = 2
output_dim = 2

# Steady   or Unsteady
# Elliptic or Parabolic
is_unsteady = False
input_dim = x_dim + is_unsteady

L_height = 0.5
v0 = 5.0

# Is there a true solution
exists_analytic_sol = False
# If there is a true solution, provide contour levels
plot_levels = np.linspace(-1,1,100)


def true_solution(X):
    pass


################# PDE Coefficients ########################

# PDE type:
pde_type = 'NavierStokes'

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

def bdry_con(X):
    u = torch.zeros( (X.size(0), output_dim), device=X.device)
    return u

def inlet_con(X):
    u = torch.zeros_like(X, device=X.device)
    
    u[:,0] = v0*torch.mul((L_height - X[:,1]),(L_height + X[:,1]))

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


boundingbox = [ [0, 5*L_height], [-L_height,L_height] ]

centre1 = [L_height,0]
radius1 = L_height/3

disk1 = [ 'disk', centre1, radius1, bdry_con ]

inlet_left =   [ 'line', [0, -L_height], [1,0], [ [0, -L_height], [0, L_height] ], inlet_con]
wall_top =     [ 'line', [0, L_height], [0,-1], [ [0, L_height], [5*L_height, L_height] ], bdry_con]
wall_bot =     [ 'line', [0,-L_height], [0, 1], [ [0, -L_height], [5*L_height, -L_height] ], bdry_con]
outlet_right = [ 'line', [5*L_height, -L_height], [-1,0], [ [5*L_height, -L_height], [5*L_height, L_height] ], None ]



my_bdry = [ disk1, inlet_left, wall_top, wall_bot, outlet_right ]

is_periodic = False