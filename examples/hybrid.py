###
### Test the domain decomposition problem
###
### Laplace's equation, solution 4*(x-1)^2 - 4*(y+0.5)^2
###
### Region bounded by polar curve r(theta) = 0.75*cos(theta) + 1.1
###        In a rectangle [0,1] x [-1,1], use a finite difference solver
###        Place walkers outside the square
###        Parameter for updating finite difference solution 
###

import torch
import math
import numpy as np

############## Save model and/or Load model ##############

savemodel = ''
loadmodel = ''

############## Problem Parameters ################

# Physical Dimension
x_dim = 2
output_dim = 1

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


############### PDE Coefficients ########################

# PDE type:
#     NavierStokes, StokesFlow, viscousBurgers, Laplace, Elliptic, Parabolic
pde_type = 'Laplace'

# Diffusion coefficient
def diffusion(X):
    mu = torch.tensor( 1.0 )
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
    ubdry = 4.0*( X[:,0] - 1.0 )**2 - 4.0*( X[:,1] + 0.5 )**2
    return ubdry[:,None]

def init_con(X):
    pass


#################  Make the domain  #######################
###
### See doc/param_problem.md for usage details
###
### TODO Fix

### WARNING: Use decimals to make sure torch defaults to floating point

boundingbox = [ [-0.5, 2.0], [-1.5, 1.5] ]

def polar_eq(theta):
    r = 0.75*torch.cos(theta) + 1.1
    return r

polar = {'type':'polar',
         'equation': polar_eq,
         'boundary_condition': bdry_con}

box = { 'type': 'box',
        'xinterval': [ 0.0, 1.0],
        'yinterval': [-1.0, 1.0]}

list_of_walls = [polar]
list_of_periodic_ends =[]
solid_walls = []
inlet_outlet = []
mesh = [box]
