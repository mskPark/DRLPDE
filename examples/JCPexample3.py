###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 3: Stokes Flow in Sphere
###

import torch
import math
import numpy as np

############## Save model and/or Load model ##############

savemodel = 'JCPexample3'
loadmodel = ''

# Physical Dimension
x_dim = 3
output_dim = 3

# Steady   or Unsteady
# Elliptic or Parabolic
is_unsteady = False
input_dim = x_dim + is_unsteady

# Is there a true solution
exists_analytic_sol = True
# If there is a true solution, provide contour levels
plot_levels = np.linspace(-1,1,100)

def true_solution(X):
    u = torch.stack( ( -X[:,1], X[:,0], torch.zeros(X.size(0), device=X.device )), dim=1)
    return u


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


boundingbox = [ [-1,1], [-1,1], [-1,1] ]

centre1 = [0,0,0]
radius1 = 1.0

bdry1 = [ 'ball3D', centre1, radius1, bdry_con ]

sphere1 = {   'type':'sphere',
            'centre': [0,0,0],
            'radius': 1.0,
            'boundary_condition':bdry_con }
            
list_of_dirichlet_boundaries = [sphere1]
list_of_periodic_boundaries =[]