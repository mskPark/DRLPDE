###
### This module contains the classes to create the domain and points
###

import torch
import math
import numpy as np
import DRLPDE.bdry as bdry
import DRLPDE.bdry3d as bdry3d

### TODO list ###

# Space Domain made as the intersection of regions
# TODO 1) union of regions
# 
#  TODO 2) Should the dimension d include the time/hyperparameter dimension?
#   Typically the range in hyperparameters is large (sometimes on the exponential scale)
#   For each hyperparameter added

# TODO 3) Scale number of initial condition points by (l/V)^(d/2)
###        l = space measure
###        V = space measure * time
###        d = input_dim[0]

# TODO 4) Fix number of points generated on wall
#         Currently, if the wall length is < 1, then we need more points.

# TODO 5) More than one set of mesh points
#         Fix in main.py

### Space Domain D in R^{d}, d = 2 or 3

class theDomain:

    def __init__(self, boundingbox, list_of_walls, solid_walls, inlet_outlet, list_of_periodic_ends, mesh):
        
        self.boundingbox = boundingbox

        # Dirichlet boundary
        self.wall = []
        self.wall_bdry = []
        for specs in list_of_walls:
            ### 2D walls
            if specs['type'] == 'line':
                self.wall.append( bdry.line( endpoints = specs['endpoints'],
                                                      normal = specs['normal'],
                                                      bc = specs['boundary_condition'] ))
            
            if specs['type'] == 'circle':
                self.wall.append( bdry.circle( centre = specs['centre'],
                                                      radius = specs['radius'], 
                                                      bc = specs['boundary_condition'] ))

            if specs['type'] == 'ring':
                self.wall.append( bdry.ring( centre = specs['centre'],
                                                      radius = specs['radius'], 
                                                      bc = specs['boundary_condition'] ))
            
            if specs['type'] == 'polar':
                self.wall.append( bdry.polar( polar_eq = specs['equation'],
                                                bc = specs['boundary_condition']))

            ### 3D walls - Note: 2D and 3D  walls not compatible with each other
            if specs['type'] == 'sphere':
                self.wall.append( bdry3d.sphere( centre = specs['centre'], 
                                                     radius = specs['radius'], 
                                                     bc = specs['boundary_condition']))
            
            if specs['type'] == 'ball':
                self.wall.append( bdry3d.ball( centre = specs['centre'],
                                                   radius = specs['radius'],
                                                   bc = specs['boundary_condition']))
            
            if specs['type'] == 'cylinder':
                self.wall.append( bdry3d.cylinder( centre = specs['centre'],
                                                       radius = specs['radius'],
                                                       ### TODO axis of rotation
                                                       bc = specs['boundary_condition'] ))
            
            if specs['type'] == 'plane':
                self.wall.append( bdry3d.plane( point = specs['point'],
                                                    normal = specs['normal'],
                                                    corners = specs['corners'],
                                                    bc = specs['boundary_condition']))

        # Periodic ends
        self.periodic = []
        for specs in list_of_periodic_ends:
            self.periodic.append( bdry_periodic( variable = specs['variable'],
                                                 bot = specs['bot'],
                                                 top = specs['top']  ))

        # Inlets and outlets
        self.inletoutlet = []
        self.inletoutlet_bdry = []
        for specs in inlet_outlet:
            ### 2D inlet/outlet
            if specs['type'] == 'line':
                self.inletoutlet.append( bdry.line( endpoints = specs['endpoints'],
                                                      normal = specs['normal'],
                                                      bc = specs['boundary_condition'] ))
            
            ### 3D inlet/outlet
            if specs['type'] == 'plane':
                self.inletoutlet.append( bdry3d.plane( point = specs['point'],
                                                    normal = specs['normal'],
                                                    corners = specs['corners'],
                                                    bc = specs['boundary_condition']))

        # Solid boundary
        self.solid = []
        for specs in solid_walls:
            if specs['type'] == 'disk':
                self.solid.append( bdry.disk( centre = specs['centre'],
                                                      radius = specs['radius'],
                                                      bc = specs['boundary_condition'] ))
            
            if specs['type'] == 'solidball':
                self.solid.append( bdry3d.ball( centre = specs['centre'],
                                                   radius = specs['radius'],
                                                   bc = specs['boundary_condition']))
            
        # Mesh
        self.mesh = []
        for specs in mesh:
            if specs['type'] == 'grid':
                    self.mesh.append( bdry.box(xint = specs['xinterval'],
                                                 yint = specs['yinterval']))

        # Walker boundary. If walker crosses, flag and bring to boundary, evaluate BC
        self.boundary = self.wall + self.inletoutlet

        # For checking whether points are inside/outside the domain
        self.inside = self.wall + self.inletoutlet + self.mesh

        # Calculate volume of domain
        # Change the number of digits required
        self.volume = self.volumeDomain(1e-3)

    ### Calculate volume of the domain
    def volumeDomain(self, std):
        # Approximates volume of domain through Monte Carlo Integration
        # Sample from boundingbox B, estimate volume of domain D as
        # vol(D) = vol(B)*(fraction of samples inside D)
        # std of estimate ~ vol(B) / sqrt(N)

        volB = 1

        # Calculate volume of boundingbox
        for ii in range(len(self.boundingbox)):
            volB = volB*(self.boundingbox[ii][1] - self.boundingbox[ii][0])

        # Calculate number needed to get std within tol (approximate)
        num = np.int( (volB/std)**2 ) 
        X = torch.empty( (num, len(self.boundingbox)) )

        # Uniformly sample from boundingbox
        for ii in range(len(self.boundingbox)):
            X[:,ii] = (self.boundingbox[ii][1] - self.boundingbox[ii][0])*torch.rand( (num) ) + self.boundingbox[ii][0]

        outside = torch.zeros( X.size(0), dtype=torch.bool)
        for wall in self.boundary + self.mesh:
            outside += wall.distance(X) < 0
        
        frac = (num - torch.sum(outside))/num
        volD = volB*frac

        #standard_error = torch.sqrt( volB**2 ( 1 - frac)*frac/num/(num-1) )

        # 95% confident that the answer = VolD +- standard_error*1.96 
        # TODO: If 2*1.96*standard_error > 0.01 (For 2 digit accuracy, 95% confidence)
        #       Redo, and use the previous answer to double the number of points.
        #       Repeat until we have desired accuracy.


        return volD

### Periodic Boundary class
class bdry_periodic:
    
    ### index
    ### bot
    ### top

    def __init__(self, variable, bot, top):
        
        if variable == 'x':
            self.index = 0
        if variable == 'y':
            self.index = 1
        if variable == 'z':
            self.index = 2

        self.bot = bot
        self.top = top

class thePoints:

    def __init__(self):

        self.toTrain = [InteriorPoints(num, domain, input_dim, input_range)]
        self.target = [interior_target]
        self.L2optimizers = [torch.optim.Adam()]
        self.Linfoptimizers = [torch.optim.Adam()]

        for bdry in domain.bdry:
            self.toTrain.append()
            self.target.append()
            self.L2optimizers.append()
            self.Linfoptimizers.append()
        
        for inletoutlet in domain.inletoutlet:
            self.toTrain.append()
            self.target.append()
            self.L2optimizers.append()
            self.Linfoptimizers.append()
        
        for mesh in domain.mesh:
            self.toTrain.append()
            self.target.append()
            self.L2optimizers.append()
            self.Linfoptimizers.append()

        if unsteady:
            self.toTrain.append()
            self.target.append()
            self.L2optimizers.append()
            self.Linfoptimizers.append()

        # Points.toTrain = [Interior, Bdry, ... , Bdry, IC]
        # Points.target = [target, BC, ... , BC, IC]
        # Points.L2optimizers = [op1, ...., opx]
        # Points.Linfoptimizers = [op1, ..., opx]
        # Points.error = [] or [interior, bdry, ..., bdry, IC]

### DataLoader classes for different types of points

class InteriorPoints(torch.utils.data.Dataset):
    ###
    ### Points that are inside the domain
    ###

    def __init__(self, num, domain, input_dim, input_range):
        
        self.location = generate_interior_points(num, input_dim, input_range, domain, domain.inside)
        self.num_pts = num
    
    ### Required def for Dataset class
    def __len__(self):
        # How many data points are there?
        return self.num_pts
    
    def __getitem__(self, index):
        # Retrieves one sample of data
        return self.location[index,:], index
        
class BCPoints(torch.utils.data.Dataset):
    ###
    ### Points that are along the (d-1) dimensional boundary of the domain
    ### Evaluate the Dirichlet boundary condition at these points
    ###
    
    def __init__(self, num, domain, input_dim, input_range):
        
        Xbdry, Ubdry = generate_boundary_points(num, input_dim, input_range, domain.wall)
        
        self.location = Xbdry
        self.num_pts = num
        self.value = Ubdry
        
    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, index):
        return self.location[index,:], self.value[index,:], index    
    
class ICPoints(torch.utils.data.Dataset):
    ###
    ### Points that are along the initial time of the space-time domain
    ### Evaluate the initial condition at these points
    ###

    def __init__(self, num_init, input_dim, input_range, domain, init_con):
        
        Xinit = generate_initial_points(num_init, input_dim, input_range, domain)
        
        self.location = Xinit
        self.num_pts = num_init
        self.value = init_con(Xinit)
        
    def __len__(self):
        ### How many data points are there?
        return self.num_pts
    
    def __getitem__(self, index):
        ### Gets one sample of data
        ### 
        return self.location[index,:], self.value[index,:], index

class InletOutletPoints(torch.utils.data.Dataset):
    ### At inlets/outlets
    ### Pressure specified & velocity x (normal vector to inletoutlet) specified
    ### or
    ### Gradient of pressure & velocity x (normal vector to inletout) specified
    def __init__(self, num, domain, input_dim, input_range):
        
        Xbdry, Ubdry = generate_boundary_points(num, input_dim, input_range, domain.inletoutlet)
        
        self.location = Xbdry
        self.num_pts = num
        self.value = Ubdry
        
    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, index):
        return self.location[index,:], self.value[index,:], index    

class SolidWallPoints(torch.utils.data.Dataset):
    ### Solid walls require a different number of points
    ###   It seems easier to separate these d dimensional boundary from the (d-1) dimensional boundary
    ### Do not need to check for exits on them
    ###   Problem should be setup so that there is a BCpoints class for the edge

    def __init__(self, num, domain, input_dim, input_range):
        
        Xbdry, Ubdry = generate_boundary_points(num, input_dim, input_range, domain.solid)
        
        self.location = Xbdry
        self.num_pts = num
        self.value = Ubdry
        
    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, index):
        return self.location[index,:], self.value[index,:], index    

class MeshPoints(torch.utils.data.Dataset):
    # Can only do 1 meshgrid
    # x_interval and y_interval has to be the same length
    # 
    # TODO: Be able to handle multiple meshgrids
    # TODO: Refinement study

    def __init__(self, num, box, model, dev):

        #TODO Compute laplace operator once, save as SVD

        X = torch.cartesian_prod( torch.linspace(box.xint[0], box.xint[1], num)[1:-1], 
                                  torch.linspace(box.yint[0], box.yint[1], num)[1:-1] )
        
        self.A = box.make_laplace(num).to(dev)
        b, Xwall = box.rhs(num, model, dev)

        U = self.solveU(b)

        self.Xwall = Xwall
        self.location = X
        self.num_pts = X.size(0)
        self.value =  U
    
    def solveU(self, b):
        U = torch.linalg.solve(self.A, b)
        return U

    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, index):
        return self.location[index,:], self.value[index,:], index

### Number of points along boundary
###   based on Mean Minimum Distance
def num_points_wall(N, l, V, d, d0):

    Nw = int( 4*torch.round( (l/V)**(d0/2) * N**(d0/d) ).detach().numpy() + 1 )

    return Nw

### Calculate how many points to make on each boundary type from number of interior points (num)
def numBCpoints(num, input_dim, domain, bdry):

    # TODO 2)
    dim = input_dim[0]

    wall_total_measure = 0
    
    for wall in bdry:
        wall_total_measure += wall.measure
        wall_dim = wall.dim
    num_bc = num_points_wall(num, wall_total_measure, domain.volume, dim, wall_dim)

    return num_bc

def numICpoints(num, input_dim):
    ### TODO: Scale by (l/V)^(d/2)
    ###        l = space measure
    ###        V = space measure * time
    ###        d = input_dim[0]

    num_ic = int( num**( input_dim[0]/(input_dim[0] + input_dim[1]) ) )

    return num_ic

###
### Functions to generate points
### 

def generate_interior_points(num, input_dim, input_range, domain, within_domain):
    ### Generate points inside the domain

    X = torch.zeros( ( num, int(np.sum(input_dim)) ) )

    for ii in range(input_dim[0]):
        X[:,ii] = (input_range[ii][1] - input_range[ii][0])*torch.rand( (num) ) + input_range[ii][0]

    outside = torch.zeros( X.size(0), dtype=torch.bool)
    for wall in within_domain:
        outside += wall.distance(X) < 0

    if any(outside):
        X[outside,:] = generate_interior_points(torch.sum(outside), input_dim, input_range, domain, within_domain)
        
    if input_dim[1]:
        # Fill in time values
        t = input_dim[0]
        X[:,t] = (input_range[t][1] - input_range[t][0])*torch.rand( (num) ) + input_range[t][0]

    if input_dim[2]:
        # Fill in hyperparameter values
        # TODO Do exponential scaling
        hstart = input_dim[0] + input_dim[1]
        hend = sum(input_dim)
        for jj in range(hstart, hend):
            X[:,jj] = (input_range[jj][1] - input_range[jj][0])*torch.rand( (num) ) + input_range[jj][0]

    return X

def generate_boundary_points(num, input_dim, input_range, bdry):

    # num = number of points needed along boundary
    num_walls = len(bdry)

    points_per_wall = []
    utrue_per_wall = []

    # TODO 4)
    ### How many points to make on each wall
    for ii in range(num_walls):
        wall = bdry[ii]
        num_points_wall = int(num*np.maximum( wall.measure, 1.0))
        Xwall = torch.zeros( ( num_points_wall, int(np.sum(input_dim))) )
        Xwall[:,:input_dim[0]] = wall.make_points(num_points_wall)

        # Include time
        if input_dim[1]:
            # Fill in time values
            t = input_dim[0]
            Xwall[:,t] = (input_range[t][1] - input_range[t][0])*torch.rand( (num_points_wall) ) + input_range[t][0]
        # Include hyperparameters
        if input_dim[2]:
            # Fill in hyperparameter values
            # TODO Do exponential scaling
            hstart = input_dim[0] + input_dim[1]
            hend = sum(input_dim)
            for jj in range(hstart, hend):
                Xwall[:,jj] = (input_range[jj][1] - input_range[jj][0])*torch.rand( (num_points_wall) ) + input_range[jj][0]

        # Evaluate boundary condition
        Uwall = wall.bc(Xwall)

        points_per_wall.append( Xwall )
        utrue_per_wall.append( Uwall )

    Xbdry = torch.cat( points_per_wall, dim=0)
    Ubdry = torch.cat( utrue_per_wall, dim=0)

    # Sample from above boundary points
    indices = torch.multinomial( torch.arange( Xbdry.size(0), dtype=torch.float ), num)
    
    Xbdry = Xbdry[indices,:]
    Ubdry = Ubdry[indices,:]
    
    return Xbdry, Ubdry

def generate_initial_points(num, input_dim, input_range, domain):
    ### Generate points for initial condition

    X = generate_interior_points(num, input_dim, input_range, domain)

    # Zero out time values
    t = input_dim[0]
    X[:,t] = torch.zeros( (num) )

    return X


