###
### Journal of Computational Physics Submission
###     Deep Reinforcement Learning of Viscous Incompressible Flow
###     Example 2: Unsteady Stokes Flow in Disk
###

import torch
import math
import numpy as np

import scipy.special as bessel

############## Save model and/or Load model ##############

savemodel = 'JCPexample2'
loadmodel = ''

# Physical Dimension
x_dim = 2
output_dim = 2

# Steady   or Unsteady
# Elliptic or Parabolic
is_unsteady = True
input_dim = x_dim + is_unsteady

# Give the time range
if is_unsteady:
    time_range = [0.0, 0.25]

# True solution
exists_analytic_sol = True
# If there is a true solution, provide contour levels
plot_levels = np.linspace(-1,1,100)

def true_solution(X):
    
    Truncate = 20
    roots = bessel.jn_zeros(1,Truncate)
    
    c = np.zeros(Truncate)
    
    u = torch.stack( ( -X[:,1], X[:,0] ), dim=1)
    
    r = torch.norm(X, dim=1)
    th = np.angle( np.complex( X[:,0].detach().cpu().numpy(), X[:,1].detach().cpu().numpy() ) )
    
    for jj in range(Truncate):
        c[jj] = 2*(-bessel.j0(roots[jj])/roots[jj])/(bessel.jv(2,roots[jj])**2)
        u[:,0] += v0*c[jj]*torch.sin(th)*torch.tensor(bessel.j1(roots[jj]*r.detach().cpu().numpy()), device=X.device)*torch.exp(-mu*roots[jj]**2*X[:,2])
        u[:,1] += -v0*c[jj]*torch.cos(th)*torch.tensor(bessel.j1(roots[jj]*r.detach().cpu().numpy()), device=X.device)*torch.exp(-mu*roots[jj]**2*X[:,2])
                  
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
#
# To make different boundary conditions for each boundary
#     ensure the correct bdry_con is called when defining the boundaries

def bdry_con(X):
    u = torch.stack( ( -X[:,1], X[:,0] ), dim=1)
    return u

def init_con(X):
    u = torch.zeros( ( X.size(0),output_dim), device=X.device)
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


boundingbox = [ [-1,1], [-1,1] ]

centre1 = [0,0]
radius1 = 1.0
endpoints1 = []

bdry1 = [ 'ring', centre1, radius1, endpoints1, bdry_con ]

my_bdry = [ bdry1 ]

is_periodic = False
