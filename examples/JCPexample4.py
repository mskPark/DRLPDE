###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 4: Poiseuille Flow
###

import torch
import math
import numpy as np

global pressure_constant, mu, cylinder_radius

############## Global variables ###################

# Pressure Gradient -> Negative to induce downward movement
pressure_constant = -5

cylinder_radius = 1.0

############## Save model and/or Load model ##############

savemodel = 'JCPexample4'
loadmodel = ''

# Physical Dimension
x_dim = 3
output_dim = 3

# Steady   or Unsteady
# Elliptic or Parabolic
is_unsteady = False
input_dim = x_dim + is_unsteady

################# Analytic Solution ######################

exists_analytic_sol = True
# If there is a true solution, provide contour levels
plot_levels = np.linspace(-1,1,100)

def true_solution(X):
    u = torch.stack( ( torch.zeros(X.size(0), device=X.device),
                       torch.zeros(X.size(0), device=X.device),
                       pressure_constant/4/mu*(X[:,0]**2 + X[:,1]**2 - cylinder_radius) ), dim=1)
    return u


################# PDE Coefficients ########################

# PDE type:
pde_type = 'NavierStokes'

# Diffusion coefficient
mu = 1

# Forcing term
def forcing(X):
    f = torch.zeros( (X.size(0), output_dim), device=X.device)
    f[:,2] = pressure_constant

    return f

################# Boundary and Initial Conditions ###########
# Use pytorch expressions to make boundary and initial conditions 
#
# To make different boundary conditions for each boundary
#     ensure the correct bdry_con is called when defining the boundaries

def bdry_con(X):
    u = torch.zeros( (X.size(0), output_dim), device=X.device)
    return u

#################  Make the domain  #######################

boundingbox = [ [-1,1], [-1,1], [0,1] ]

periodic1 = { 'variable':'z', 
              'base':0,
              'top':1 }


cylinder1 = {'type':'cylinder',
            'centre': [0,0,0],
            'radius': 1.0,
            'boundary_condition':bdry_con }
            
list_of_dirichlet_boundaries = [cylinder1]
list_of_periodic_boundaries =[periodic1]
