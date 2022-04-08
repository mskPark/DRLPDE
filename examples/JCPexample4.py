###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 4: Poiseuille Flow
###

import torch
import math
import numpy as np

global pressure_constant, mu

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
                       pressure_constant/4/mu*(X[:,0]**2 + X[:,1]**2 - 1) ), dim=1)
    return u


################# PDE Coefficients ########################

# PDE type:
pde_type = 'NavierStokes'

# Diffusion coefficient
mu = 1

# Pressure Gradient -> Negative to induce downward movement
pressure_constant = -5

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
#     First define a bounding box containing your domain
#         Syntax: [ x interval, y interval, z interval ]
#         Points will be sampled through rejection sampling
#
#     Define each boundary
#         lines: [ 'line', point, normal, endpoints, bdry_condition ]
#         disk:  [ 'disk', centre, radius, endpoints, bdry_condition]

boundingbox = [ [-1,1], [-1,1], [0,1] ]

cylinder1 = [ 'cylinder', [0,0], 1.0, bdry_con ]
plane_bot = [ 'plane', [0,0,0], [0,0,1], [ [-1, -1, 0], [1,1,0] ], None ]
plane_top = [ 'plane', [0,0,1], [0,0,-1], [ [-1,-1, 1], [1,1,1] ],  None ]

my_bdry = [ cylinder1, plane_bot, plane_top ]

