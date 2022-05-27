###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 7: Taylor Green Vortex

### u(x,y,t) = cos(x) sin(y) e^{-2 \nu t}
### v(x,y,t) = -sin(x) cos(y) e^{-2 \nu t}
### p(x,y,t) = - \rho/4 (cos(2x) + cos(2y)) e^{-4\nu t}

import torch
import math
import numpy as np

global mu

############## Global variables ###################

mu = 1

############## Save model and/or Load model ##############

savemodel = 'TaylorGreen'
loadmodel = ''

# Physical Dimension
x_dim = 2
output_dim = 2

# Steady   or Unsteady
# Elliptic or Parabolic
is_unsteady = True
input_dim = x_dim + is_unsteady
time_range = [0,0.25]

################# Analytic Solution ######################

exists_analytic_sol = True
# If there is a true solution, provide contour levels
plot_levels = np.linspace(-1,1,100)

def true_solution(X):
    u = torch.stack( ( torch.cos(X[:,0])*torch.sin(X[:,1])*torch.exp(-2*mu*X[:,2]),
                       -torch.sin(X[:,0])*torch.cos(X[:,1])*torch.exp(-2*mu*X[:,2]) ), dim=1)
    return u


################# PDE Coefficients ########################

# PDE type:
pde_type = 'NavierStokes'

# Diffusion coefficient
# mu = 1

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

def init_con(X):
    u = torch.stack( ( torch.cos(X[:,0])*torch.sin(X[:,1]),
                       -torch.sin(X[:,0])*torch.cos(X[:,1]) ), dim=1)
    return u

#################  Make the domain  #######################

boundingbox = [ [-math.pi, math.pi], [-math.pi,math.pi], ]

periodic1 = { 'variable':'x', 
              'base':-math.pi,
              'top':math.pi }

periodic2 = { 'variable':'y', 
              'base':-math.pi,
              'top':math.pi }

list_of_dirichlet_boundaries = []
list_of_periodic_boundaries =[periodic1, periodic2]
