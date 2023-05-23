###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 3: Stokes Flow in Sphere
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
        u = torch.stack( ( -X[:,1], X[:,0], torch.zeros(X.size(0), device=X.device )), dim=1)
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
#
# To make different boundary conditions for each boundary
#     ensure the correct bdry_con is called when defining the boundaries

def bdry_con(X):
    u = torch.stack( ( -X[:,1], X[:,0], torch.zeros(X.size(0), device=X.device )), dim=1)
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


boundingbox = [ [-1.0 ,1.0 ], [-1.0 ,1.0 ], [-1.0 ,1.0 ] ]

centre1 = [0.0 ,0.0 ,0.0]
radius1 = 1.0

sphere1 = { 'type':'sphere',
            'centre': centre1,
            'radius': radius1,
            'boundary_condition':bdry_con }
            
list_of_walls = [sphere1]
list_of_periodic_ends =[]
solid_walls = []
inlet_outlet = []
mesh = []