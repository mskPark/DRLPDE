###
### Example 1: Harmonic function
###                 u(x,y) = 5.1*( x - 0.87 )^2 - 5.1*( y + 0.34 )^2
###            on a polar region 
###                 r = 0.72*cos^5(theta) + 1.2
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
        ubdry = 5.1*( X[:,0] - 0.87 )**2 - 5.1*( X[:,1] + 0.34 )**2
        return ubdry[:,None]

############## Save model and/or Load model ##############

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

################# PDE Coefficients ########################

# PDE type:
#     NavierStokes, Elliptic, Parabolic
pde_type = 'Laplace'

# Diffusion coefficient
def diffusion(X):
    mu = torch.tensor( 0.1 )
    #mu = X[:,4,None]
    return mu

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
    ubdry = 5.1*( X[:,0] - 0.87 )**2 - 5.1*( X[:,1] + 0.34 )**2
    return ubdry[:,None]


#################  Make the domain  #######################

boundingbox = [ [-1.0, 2.0], [-1.5, 1.5] ]

def polar_eq(theta):
    r = 0.72*( torch.cos(theta)**5) + 1.2
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
mesh = []
