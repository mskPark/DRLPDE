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
v0 = 1.765

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
    
    u[:,0] = v0*torch.mul((L_height - X[:,1]),(L_height + X[:,1]))/(L_height**2)

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

bdry1 = {   'type':'disk',
            'centre': [L_height,0],
            'radius': L_height/3,
            'endpoints': [],
            'boundary_condition':bdry_con }

wall_left = {'type':'line',
             'point': [0, -L_height],
             'normal': [1,0],
             'endpoints': [ [0, -L_height], [0, L_height] ],
             'boundary_condition': inlet_con }

wall_top = { 'type':'line',
             'point': [0, L_height],
             'normal':  [0,-1],
             'endpoints': [ [0, L_height], [5*L_height, L_height] ],
             'boundary_condition': bdry_con }

wall_bot = {'type':'line',
             'point': [0,-L_height],
             'normal': [0, 1],
             'endpoints': [ [0, -L_height], [5*L_height, -L_height] ],
             'boundary_condition': bdry_con }

list_of_dirichlet_boundaries = [bdry1, wall_left, wall_top, wall_bot ]
list_of_periodic_boundaries =[]