###
### Unsteady Flow Past Disk with Navier-Stokes network
### 

import torch
import math
import numpy as np

collect_error = False

# Physical Dimension
x_dim = 2
output_dim = 3 #(u,v,p)

# Steady or Unsteady
t_dim = 1
if t_dim:
    t_range = [[0.0, 1.0]]
else:
    t_range = [ [] ]

# Hyperparameters
hyper_dim = 0
if hyper_dim:
    hyper_range = [[1.0, 10.0], [0.1, 1.0]]
else:
    hyper_range = [ [] ]

pressure_in = 5.0
pressure_out = -5.0
L_height = 0.5

################# PDE Coefficients ########################

# PDE type:
pde_type = 'NavierStokes'

# Diffusion coefficient
def diffusion(X):
    mu = torch.tensor(1.0)
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
    u = torch.zeros( (X.size(0), output_dim), device=X.device)
    return u

def inlet_con(X):
    u = torch.zeros((X.size(0), 1), device=X.device)
    u[:,0] = pressure_in*torch.tanh(X[:,2])
    return u

def outlet_con(X):
    u = torch.zeros((X.size(0), 1), device=X.device)
    u[:,0] = pressure_out*torch.tanh(X[:,2])
    return u

def init_con(X):
    u = torch.zeros( (X.size(0), output_dim), device=X.device)
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


boundingbox = [ [0.0, 5*L_height], [-L_height,L_height] ]

disk1 = {   'type':'disk',
            'centre': [L_height,0.0],
            'radius': L_height/3.0,
            'endpoints': [],
            'boundary_condition':bdry_con }

circle1 = {  'type':'circle',
             'centre': [L_height,0.0],
             'radius': L_height/3.0,
             'boundary_condition':bdry_con}

wall_left = {'type':'line',
             'point': [0.0, -L_height],
             'normal': [1.0, 0.0],
             'endpoints': [ [0.0, -L_height], [0.0, L_height] ],
             'boundary_condition': inlet_con }

wall_top = { 'type':'line',
             'point': [0.0, L_height],
             'normal':  [0.0, -1.0],
             'endpoints': [ [0.0, L_height], [5*L_height, L_height] ],
             'boundary_condition': bdry_con }

wall_bot = {'type':'line',
             'point': [0.0, -L_height],
             'normal': [0.0, 1.0],
             'endpoints': [ [0.0, -L_height], [5.0*L_height, -L_height] ],
             'boundary_condition': bdry_con }

wall_right = {'type':'line',
             'point': [5.0*L_height, -L_height],
             'normal': [-1.0, 0.0],
             'endpoints': [ [5.0*L_height, -L_height], [5.0*L_height, L_height] ],
             'boundary_condition': inlet_con }


list_of_walls = [circle1, wall_top, wall_bot]
list_of_periodic_ends =[]
solid_walls = [disk1]
inlet_outlet = [wall_left, wall_right]
mesh = []
