###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 1: Stokes Flow in Disk
###

import torch
import math
import numpy as np

############## Save model and/or Load model ##############

savemodel = 'JCPexample1'
loadmodel = ''

# Physical Dimension
x_dim = 2
output_dim = 2

# Steady   or Unsteady
is_unsteady = False
input_dim = x_dim + is_unsteady

# Is there a true solution
exists_analytic_sol = True
# If there is a true solution, provide contour levels
plot_levels = np.linspace(-1,1,100)

def true_solution(X):
    u = torch.stack( ( -X[:,1], X[:,0] ), dim=1)
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

def bdry_con(X):
    u = torch.stack( ( -X[:,1], X[:,0] ), dim=1)
    return u

#################  Make the domain  #######################

boundingbox = [ [-1,1], [-1,1] ]

ring1 = {   'type':'ring',
            'centre': [0,0],
            'radius': 1.0,
            'endpoints': [],
            'boundary_condition':bdry_con }

list_of_dirichlet_boundaries = [ring1]
list_of_periodic_boundaries =[]
