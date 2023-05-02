###
### Default Problem Parameters
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
        u = torch.zeros( (X.size(0), output_dim), device=X.device )
        return u

############## Problem Parameters ################

# Physical Dimension
x_dim = 2
output_dim = 2

# Steady or Unsteady
t_dim = 1
if t_dim:
    t_range = [[0.0, 1.0]]
else:
    t_range = [ [] ]

# Hyperparameters
hyper_dim = 2
if hyper_dim:
    hyper_range = [[0.0, 1.0], [1.0, 5.0]]
else:
    hyper_range = [ [] ]

L_height = 0.5

############### PDE Coefficients ########################

# PDE type:
#     NavierStokes, StokesFlow, viscousBurgers, Elliptic, Parabolic
pde_type = 'viscousBurgers'

# Diffusion coefficient
def diffusion(X):
    #mu = torch.tensor( 0.1 )
    mu = X[:,4,None]
    return mu

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
###
### Use pytorch expressions to make boundary and initial conditions 
###
### Function can use hyperparameters that are incorporated in X

def bdry_con(X):
    u = torch.zeros( (X.size(0), output_dim), device=X.device )
    return u

def init_con(X):
    u = torch.zeros( (X.size(0), output_dim), device=X.device )
    return u

def inlet_con(X):
    u = torch.zeros((X.size(0), output_dim), device=X.device)
    u[:,0] = X[:,3]*(L_height - X[:,1])*(L_height + X[:,1])/(L_height**2)
    return u

def inletoutlet_con(X):
    pass

#################  Make the domain  #######################

### WARNING: Use decimals to make sure pytorch defaults to floating point

boundingbox = [ [0, 5*L_height], [-L_height,L_height] ]

disk1 = {   'type':'disk',
            'centre': [L_height,0],
            'radius': L_height/3,
            'boundary_condition':bdry_con }

circle1 = {  'type':'circle',
             'centre': [L_height,0],
             'radius': L_height/3,
             'boundary_condition':bdry_con}

wall_left = {'type':'line',
             'normal': [1.0, 0.0],
             'endpoints': [ [0.0, -L_height], [0.0, L_height] ],
             'boundary_condition': inlet_con }

wall_top = { 'type':'line',
             'normal':  [0.0,-1.0],
             'endpoints': [ [0.0, L_height], [5.0*L_height, L_height] ],
             'boundary_condition': bdry_con }

wall_bot = {'type':'line',
             'normal': [0.0, 1.0],
             'endpoints': [ [0.0, -L_height], [5.0*L_height, -L_height] ],
             'boundary_condition': bdry_con }

wall_right = {'type':'line',
             'normal': [-1.0, 0.0],
             'endpoints': [ [5.0*L_height, -L_height], [5.0*L_height, L_height] ],
             'boundary_condition': inlet_con }



list_of_walls = [circle1, wall_left, wall_top,  wall_bot, wall_right]
list_of_periodic_ends =[]
solid_walls = [disk1]
inlet_outlet = []
mesh = []
