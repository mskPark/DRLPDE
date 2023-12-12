### Area: 0.894119
### Length: 3.7286

import torch
import math
import numpy as np

############## Collect Errors ######################

collect_error = False
num_error = 2**15
# TODO: Decide num_error automatically based on tolerance

def true_fun(X):
    ubdry = 0
    return ubdry[:,None]

############## Save model and/or Load model ##############

# Physical Dimension
x_dim = 2
output_dim = 1

# Steady or Unsteady
t_dim = 1
if t_dim:
    t_range = [[0.0, 5.0]]
else:
    t_range = [ [] ]

# Hyperparameters
hyper_dim = 0
if hyper_dim:
    hyper_range = [[]]
else:
    hyper_range = [ [] ]

################# PDE Coefficients ########################

# PDE type:
pde_type = 'Heat'

# Diffusion coefficient
def diffusion(X):
    mu = torch.tensor( 0.5 )
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
    r = X[:,0]**2 + X[:,1]**2
    ubdry = 2.0*(1 - r)*torch.cos(torch.sqrt(r))
    return ubdry[:,None]

def init_con(X):
    r = X[:,0]**2 + X[:,1]**2
    uinit = 2.0*(1 - r)*torch.cos(torch.sqrt(r))
    return uinit[:,None]

#################  Make the domain  #######################

boundingbox = [ [-1.6, 1.6], [-0.8, 0.8] ]

def polar_eq(theta):
    r = torch.cos(theta)**2 + 0.5
    return r

def dr(theta):
    dr = -2.0*torch.cos(theta)*torch.sin(theta)
    return dr

polar = {'type':'polar',
         'equation': polar_eq,
         'derivative': dr,
         'boundary_condition': bdry_con}


list_of_walls = [polar]
list_of_periodic_ends =[]
solid_walls = []
inlet_outlet = []
mesh = []
