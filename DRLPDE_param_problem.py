###
### Parameters
###

import torch
import math
import numpy as np

############## Save model and/or Load model ##############

savemodel = 'Test'
loadmodel = ''

# Physical Dimension
x_dim = 2
output_dim = 2

# Steady   or Unsteady
# Elliptic or Parabolic
is_unsteady = False
input_dim = x_dim + is_unsteady

# Give the time range
if is_unsteady:
    time_range = [0.0, 1.0]

# True solution
exists_analytic_sol = False
def true_sol(X):
    pass
    
################# PDE Coefficients ########################

# PDE type:
#     NavierStokes, Elliptic, Parabolic
#     TODO: safeguard for elliptic + is_unsteady
pde_type = 'Elliptic'

# Diffusion coefficient
mu = 0.1

# Forcing term
def forcing(X):
    f = torch.zeros( (X.size(0), output_dim), device=X.device)
    return f

# Drift coefficient for Elliptic/Parabolic PDES
def drift(X):
    drift = torch.zeros( (X.size(0), x_dim), device=X.device)
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
