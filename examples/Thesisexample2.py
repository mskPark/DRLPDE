import torch
import math
import numpy as np

import scipy.special as bessel

v0 = 4.0

############## Collect Errors ######################

collect_error = False
num_error = 2**15

# Physical Dimension
x_dim = 2
output_dim = 2

# Steady or Unsteady
t_dim = 1
if t_dim:
    t_range = [[0.0, 3.0]]
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
pde_type = 'viscousBurgers'

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

def bdry_con(X):
    u = v0*(torch.sin(math.pi*X[:,2])**2)[:,None]*torch.stack( ( -X[:,1], X[:,0] ), dim=1)
    return u

def init_con(X):
    u = torch.zeros( ( X.size(0),output_dim), device=X.device)
    return u


#################  Make the domain  #######################

boundingbox = [ [-1.0, 1.0], [-1.0, 1.0] ]

ring1 = {   'type':'ring',
            'centre': [0.0 ,0.0],
            'radius': 1.0,
            'endpoints': [],
            'boundary_condition':bdry_con }

circle2 = { 'type':'circle',
            'centre':[0.0, 0.5],
            'radius': 0.25,
            'endpoints':[],
            'boundary_condition': init_con }

disk2 = {  'type':'disk',
            'centre':[0.0, 0.5],
            'radius': 0.25,
            'endpoints':[],
            'boundary_condition': init_con }

circle3 = { 'type':'circle',
            'centre':[-0.3535, -0.3535],
            'radius': 0.25,
            'endpoints':[],
            'boundary_condition': init_con }

disk3 = {  'type':'disk',
            'centre':[-0.3535, -0.3535],
            'radius': 0.25,
            'endpoints':[],
            'boundary_condition': init_con }

circle4 = { 'type':'circle',
            'centre':[0.3535, -0.3535],
            'radius': 0.25,
            'endpoints':[],
            'boundary_condition': init_con }

disk4 = {  'type':'disk',
            'centre':[0.3535, -0.3535],
            'radius': 0.25,
            'endpoints':[],
            'boundary_condition': init_con }
            
list_of_walls = [ring1]
list_of_periodic_ends =[]
solid_walls = []
inlet_outlet = []
mesh = []