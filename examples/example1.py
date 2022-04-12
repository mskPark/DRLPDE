###
### Example 1: Laplace equation on annulus with harmonic function as boundary condition
###            u(x,y) = log(x^2 + y^2)
###            In a bounded domain not including the origin
###

import torch
import math
import numpy as np

############## Save model and/or Load model ##############

savemodel = 'example1'
loadmodel = ''

# Physical Dimension
x_dim = 2
output_dim = 1

# Steady   or Unsteady
is_unsteady = False
input_dim = x_dim + is_unsteady

# True solution
exists_analytic_sol = True
def true_solution(X):
    u = torch.log( (X[:,0]+1)**2 + X[:,1]**2 )
    return u


################# PDE Coefficients ########################

# PDE type:
#     NavierStokes, Elliptic, Parabolic
pde_type = 'Elliptic'

# Diffusion coefficient
mu = 1

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

def bdry_con(X):

    u = torch.log( ( X[:,0] + 1)**2 + X[:,1]**2 )

    return u

#################  Make the domain  #######################

boundingbox = [ [-3,3], [-2,2] ]

bdry1 = {   'type':'disk',
            'centre': [-1,0],
            'radius': 0.75,
            'endpoints': [],
            'boundary_condition':bdry_con }

bdry2 = {   'type':'line',
            'point': [-3,1],
            'normal': [-1,3],
            'endpoints': [ [-3,1], [0,2] ],
            'boundary_condition':bdry_con }

wall_left = {'type':'line',
             'point': [-3,-2],
             'normal': [-1,0],
             'endpoints': [ [-3,-2], [-3,1] ],
             'boundary_condition': bdry_con }

wall_top = { 'type':'line',
             'point': [-3,2],
             'normal': [0,1],
             'endpoints': [ [0,2],   [3,2]  ],
             'boundary_condition': bdry_con }

wall_right= {'type':'line',
             'point': [3, -2],
             'normal':  [1,0],
             'endpoints':  [ [3,0],   [3,2]  ],
             'boundary_condition': bdry_con }

wall_bot = {'type':'line',
             'point': [-3,-2],
             'normal': [1,-3],
             'endpoints': [ [-3,-2], [3,0]  ],
             'boundary_condition': bdry_con }

list_of_dirichlet_boundaries = [bdry1, bdry2, wall_left, wall_top, wall_right, wall_bot ]
list_of_periodic_boundaries =[]

