###
### This module contains the classes pertaining to making the Domain
###

### Domain is defined as the intersection of domains
### TODO: Define as union of domains
###

### TODO: Put all domain classes into their own modules
### 

import torch
import math
import numpy as np

#from DomainTypes import Bdry2D, Bdry3D

### Domain
class Domain:
    ### This class defines the domain using the parameters provided
    ### It sets up the boundingbox and each boundary

    def __init__(self, is_unsteady, boundingbox, 
                list_of_dirichlet_boundaries,
                list_of_periodic_boundaries=False):
        
        self.boundingbox = boundingbox
        self.is_unsteady = is_unsteady

        self.num_of_boundaries = len(list_of_dirichlet_boundaries)
        
        # Unpack dirichlet boundary descriptions
        self.boundaries = []
        for specs in list_of_dirichlet_boundaries:
            ### 2D boundaries
            if specs['type'] == 'line':
                self.boundaries.append( bdry_line( point = specs['point'], 
                                                   normal = specs['normal'],
                                                   endpoints = specs['endpoints'],
                                                   boundary_condition = specs['boundary_condition'] ))
            
            if specs['type'] == 'disk':
                self.boundaries.append( bdry_disk( centre = specs['centre'],
                                                   radius = specs['radius'],
                                                   endpoints = specs['endpoints'], 
                                                   boundary_condition = specs['boundary_condition'] ))
            
            if specs['type'] == 'ring':
                self.boundaries.append( bdry_ring( centre = specs['centre'],
                                                   radius = specs['radius'],
                                                   endpoints = specs['endpoints'], 
                                                   boundary_condition = specs['boundary_condition'] ))
            
            ### 3D boundaries - Note: 2D and 3D boundaries not compatible with each other
            if specs['type'] == 'sphere':
                self.boundaries.append( bdry_sphere( centre = specs['centre'], 
                                                     radius = specs['radius'], 
                                                     boundary_condition = specs['boundary_condition']))
            
            if specs['type'] == 'ball':
                self.boundaries.append( bdry_ball( centre = specs['centre'],
                                                   radius = specs['radius'],
                                                   boundary_condition = specs['boundary_condition']))
            
            if specs['type'] == 'cylinder':
                self.boundaries.append( bdry_cylinder( centre = specs['centre'],
                                                       radius = specs['radius'],
                                                       ### TODO axis of rotation
                                                       boundary_condition = specs['boundary_condition'] ))
            
            if specs['type'] == 'plane':
                self.boundaries.append( bdry_plane( point = specs['point'],
                                                    normal = specs['normal'],
                                                    corners = specs['corners'],
                                                    boundary_condition = specs['boundary_condition']))

        # Unpack any periodic boundaries
        self.periodic_boundaries = []
        for specs in list_of_periodic_boundaries:
            self.periodic_boundaries.append( bdry_periodic( variable = specs['variable'],
                                                            base = specs['base'],
                                                            top = specs['top']  ))

### Boundary Classes

# 2D Boundaries
class bdry_disk:
    ### Class structure for a 2D solid disk boundary, the domain being outside the disk
    
    def __init__(self, centre, radius, endpoints, boundary_condition):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius

        self.bdry_cond = boundary_condition
        
        ### Convert endpoints to angles
        ###
        ### Use angles to sample along disk
        if len(endpoints) == 0:
            self.angles = [0, 2*math.pi]
        else:
            self.angles = []
            for point in endpoints:
                self.angles.append( np.angle( np.complex( point[0] - self.centre[0], point[1] - self.centre[1] ) ) )
            if self.angles[1] < self.angles[0]:
                self.angles[1] += 2*math.pi
            
            
    def make_bdry_pts(self, num_bdry, boundingbox, is_unsteady, other_bdrys):
        ### Make points along the boundary as well as the interior
        
        #theta = (self.angles[1] - self.angles[0])*torch.rand((2*num_bdry)) + self.angles[0]
        #inside_rad = self.radius*torch.sqrt( torch.rand(num_bdry) )

        theta_r = torch.cartesian_prod( torch.linspace(0, 2*math.pi, 2**7), self.radius*torch.sqrt( torch.linspace(0,1,2**4) ))
        #rad = self.radius*torch.sqrt( torch.linspace(0,1,2**4) )

        
        if is_unsteady:
            Xbdry1 = torch.stack((self.radius*torch.cos(theta[:num_bdry]) + self.centre[0],
                                 self.radius*torch.sin(theta[:num_bdry]) + self.centre[1],
                                (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]),dim=1 )

            Xbdry2 = torch.stack( (inside_rad*torch.cos(theta[num_bdry:]) + self.centre[0],
                                      inside_rad*torch.sin(theta[num_bdry:]) + self.centre[1],
                                     (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]),dim=1)

        else:
            #Xbdry1 = torch.stack((self.radius*torch.cos(theta[:num_bdry]) + self.centre[0],
            #                     self.radius*torch.sin(theta[:num_bdry]) + self.centre[1]), dim=1 )

            #Xbdry2 = torch.stack( (inside_rad*torch.cos(theta[num_bdry:]) + self.centre[0],
            #                         inside_rad*torch.sin(theta[num_bdry:]) + self.centre[1]), dim=1  )

            Xbdry = torch.stack( (theta_r[:,1]*torch.cos(theta_r[:,0]) + self.centre[0],
                                  theta_r[:,1]*torch.sin(theta_r[:,0]) + self.centre[1]), dim=1)

        #Xbdry = torch.cat( (Xbdry1, Xbdry2), dim=0)

        #indices = torch.multinomial( torch.arange( 2*num_bdry, dtype=torch.float ), num_bdry)
        #Xbdry = Xbdry[indices,:]


        ### Check if outside other bdrys
        ### and remake bdry points
        outside = torch.zeros(Xbdry.size(0), dtype=torch.bool)

        for bdry in other_bdrys:
            outside += bdry.dist_to_bdry(Xbdry) < 0
        
        if any(outside):
            Xbdry[outside,:] = self.make_bdry_pts(torch.sum(outside), boundingbox, is_unsteady, other_bdrys)

        return Xbdry
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        distance = ( torch.norm(X[:,:2] - self.centre.to(X.device),dim=1) - self.radius )
        return distance
    
    def plot_bdry(self, num_bdry):
        ### Give uniformly spaced points along the boundary to plot
        theta = torch.linspace(self.angles[0], self.angles[1], num_bdry)
        Xplot = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )
        
        return Xplot

class bdry_ring:
    ### Class structure for a circle boundary, the inside being the domain
    
    def __init__(self, centre, radius, endpoints, boundary_condition):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius

        self.bdry_cond = boundary_condition
        
        ### Convert endpoints to angles
        ###
        ### Use angles to sample along disk
        if len(endpoints) == 0:
            self.angles = [0, 2*math.pi]
        else:
            self.angles = []
            for point in endpoints:
                self.angles.append( np.angle( np.complex( point[0] - self.centre[0], point[1] - self.centre[1] ) ) )
            if self.angles[1] < self.angles[0]:
                self.angles[1] += 2*math.pi
            
            
    def make_bdry_pts(self, num_bdry, boundingbox, is_unsteady, other_bdrys):
        #theta = (self.angles[1] - self.angles[0])*torch.rand((num_bdry)) + self.angles[0]
        theta = torch.linspace(0, 2*math.pi, 64)

        if is_unsteady:
            #Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
            #                     self.radius*torch.sin(theta) + self.centre[1],
            #                    (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]),dim=1 )
            time_range = (boundingbox[-1][1] - boundingbox[-1][0])*torch.linspace(0,1,64) + boundingbox[-1][0]
            dummy = torch.cartesian_prod( theta, time_range )

            Xbdry = torch.stack( (self.radius*torch.cos(dummy[:,0]) + self.centre[0], 
                                 self.radius*torch.sin(dummy[:,0]) + self.centre[1],
                                 dummy[:,1]), dim=1 )

        else:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )

        ### Check if outside other bdrys
        ### and remake bdry points
        outside = torch.zeros(Xbdry.size(0), dtype=torch.bool)

        for bdry in other_bdrys:
            outside += bdry.dist_to_bdry(Xbdry) < 0
        
        if any(outside):
            Xbdry[outside,:] = self.make_bdry_pts(torch.sum(outside), boundingbox, is_unsteady, other_bdrys)
        
        return Xbdry
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        distance = (self.radius - torch.norm( X[:,:2] - self.centre.to(X.device),dim=1))

        return distance
    
    def plot_bdry(self, num_bdry):
        theta = torch.linspace(self.angles[0], self.angles[1], num_bdry)
        Xplot = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )
        
        return Xplot
       
class bdry_line:
    ### Class structure for a line boundary
    ###       normal vector points inside
    
    def __init__(self, point, normal, endpoints, boundary_condition):
        self.point = torch.tensor(  point )
        self.normal = torch.tensor( normal )
        self.constant = -sum( self.normal*self.point )
        
        self.bdry_cond = boundary_condition
        
        self.endpoints = torch.tensor(endpoints)
        
    def make_bdry_pts(self, num_bdry, boundingbox, is_unsteady, other_bdrys):
        
        if is_unsteady:
            Xbdry = torch.cat( ( (self.endpoints[1] - self.endpoints[0] )*torch.rand((num_bdry,1)) + self.endpoints[0],
                                 (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry,1)) + boundingbox[-1][0]), dim=1)
        else:    
            #Xbdry = ( self.endpoints[1] - self.endpoints[0] )*torch.rand((num_bdry,1)) + self.endpoints[0]
            Xbdry = ( self.endpoints[1] - self.endpoints[0] )*torch.linspace(0,1, 2**9)[:,None] + self.endpoints[0]
        ### Check if outside other bdrys
        ### and remake bdry points
        #outside = torch.zeros(Xbdry.size(0), dtype=torch.bool)

        #for bdry in other_bdrys:
        #    outside += bdry.dist_to_bdry(Xbdry) < 0
        
        #if any(outside):
        #    Xbdry[outside,:] = self.make_bdry_pts(torch.sum(outside), boundingbox, is_unsteady, other_bdrys)

        return Xbdry
    
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain
        distance = torch.sum( self.normal.to(X.device)*X[:,:2], dim=1) + self.constant
        
        return distance
    
    def plot_bdry(self, num_bdry):
        Xplot = ( self.endpoints[1] - self.endpoints[0] )*torch.linspace(0, 1, num_bdry)[:,None] + self.endpoints[0]
        
        return Xplot

# 3D Boundaries
class bdry_ball:
    ### Class structure for a 3D solid ball boundary, the domain being outside the ball
    
    def __init__(self, centre, radius, boundary_condition):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius

        self.bdry_cond = boundary_condition
        
            
    def make_bdry_pts(self, num_bdry, boundingbox, is_unsteady, other_bdrys):
        ### Spherical coordinates
        ###
        ### x = radius*sin(phi)*cos(theta)
        ### y = radius*sin(phi)*sin(theta)
        ### z = radius*cos(phi)

        theta = 2*math.pi*torch.rand( (num_bdry))
        phi =  math.pi*torch.rand((num_bdry))
        
        if is_unsteady:
            Xbdry = torch.stack((self.radius*torch.sin(phi)*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(phi)*torch.sin(theta) + self.centre[1],
                                 self.radius*torch.cos(phi) + self.centre[2]
                                (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]),dim=1 )  
        else:
            Xbdry = torch.stack((self.radius*torch.sin(phi)*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(phi)*torch.sin(theta) + self.centre[1],
                                 self.radius*torch.cos(phi) + self.centre[2]), dim=1 )  
        
        ### Check if outside other bdrys
        ### and remake bdry points
        outside = torch.zeros(Xbdry.size(0), dtype=torch.bool)

        for bdry in other_bdrys:
            outside += bdry.dist_to_bdry(Xbdry) < 0
        
        if any(outside):
            Xbdry[outside,:] = self.make_bdry_pts(torch.sum(outside), boundingbox, is_unsteady, other_bdrys)

        return Xbdry
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        distance = ( torch.norm( X[:,:3] - self.centre.to(X.device),dim=1) - self.radius )
        return distance
    
class bdry_sphere:
    ### Class structure for a 3D hollow sphere boundary, the domain being inside the sphere
    
    def __init__(self, centre, radius, boundary_condition):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius
        self.bdry_cond = boundary_condition
        
            
    def make_bdry_pts(self, num_bdry, boundingbox, is_unsteady, other_bdrys):
        ### Spherical coordinates
        ###
        ### x = radius*sin(phi)*cos(theta)
        ### y = radius*sin(phi)*sin(theta)
        ### z = radius*cos(phi)

        theta = 2*math.pi*torch.rand( (num_bdry))
        phi =  math.pi*torch.rand((num_bdry))
        
        if is_unsteady:
            Xbdry = torch.stack((self.radius*torch.sin(phi)*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(phi)*torch.sin(theta) + self.centre[1],
                                 self.radius*torch.cos(phi) + self.centre[2]
                                (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]),dim=1 )  
        else:
            Xbdry = torch.stack((self.radius*torch.sin(phi)*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(phi)*torch.sin(theta) + self.centre[1],
                                 self.radius*torch.cos(phi) + self.centre[2]), dim=1 )  
        
        ### Check if outside other bdrys
        ### and remake bdry points
        outside = torch.zeros(Xbdry.size(0), dtype=torch.bool)

        for bdry in other_bdrys:
            outside += bdry.dist_to_bdry(Xbdry) < 0
        
        if any(outside):
            Xbdry[outside,:] = self.make_bdry_pts(torch.sum(outside), boundingbox, is_unsteady, other_bdrys)
        
        return Xbdry
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        distance = ( self.radius - torch.norm( X[:,:3] - self.centre.to(X.device),dim=1) )
        return distance
    
class bdry_cylinder:
    ### Class structure for inside a cylindrical shell
    ### Centre: One end of the cylinder
    ### Points in z-direction
    ### TODO include axis: Points in direction, length of axis determines how long
    
    def __init__(self, centre, radius, boundary_condition):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius
        self.bdry_cond = boundary_condition
        
            
    def make_bdry_pts(self, num_bdry, boundingbox, is_unsteady, other_bdrys):
        ### Cylindrical coordinates
        ###
        ### x = radius*cos(theta)
        ### y = radius*sin(theta)
        ### z = z

        theta = 2*math.pi*torch.rand( (num_bdry))
        
        if is_unsteady:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(theta) + self.centre[1],
                                (boundingbox[-2][1] - boundingbox[-2][0])*torch.rand((num_bdry)) + boundingbox[-2][0],
                                (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]),dim=1 )  
        else:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(theta) + self.centre[1],
                                 (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]), dim=1 )  
        
        ### Check if outside other bdrys
        ### and remake bdry points
        outside = torch.zeros(Xbdry.size(0), dtype=torch.bool)

        for bdry in other_bdrys:
            outside += bdry.dist_to_bdry(Xbdry) < 0
        
        if any(outside):
            Xbdry[outside,:] = self.make_bdry_pts(torch.sum(outside), boundingbox, is_unsteady, other_bdrys)

        return Xbdry
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain
        distance = ( self.radius - torch.norm( X[:,:2] - self.centre[:2].to(X.device),dim=1) )
        return distance

class bdry_plane:
    
    ### Class structure for a plane in 3D space
    ### normal vector points inside
    ### corners should be opposite
    
    def __init__(self, point, normal, corners, boundary_condition):
        self.point = torch.tensor(  point )
        self.normal = torch.tensor( normal )
        self.constant = -sum( self.normal*self.point )
        
        self.corners = torch.tensor( corners )
        
        self.bdry_cond = boundary_condition
        
    def make_bdry_pts(self, num_bdry, boundingbox, is_unsteady, other_bdrys):
        
        ### Generate boundary points
        if is_unsteady:
            Xbdry = torch.cat( ( (self.endpoints[1] - self.endpoints[0] )*torch.rand((num_bdry,1)) + self.endpoints[0],
                                 (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry,1)) + boundingbox[-1][0]), dim=1)
        else:    
            Xbdry = ( self.corners[1] - self.corners[0] )*torch.rand((num_bdry,1)) + self.corners[0]

        ### Check if outside other bdrys
        ### and remake bdry points
        outside = torch.zeros(Xbdry.size(0), dtype=torch.bool)

        for bdry in other_bdrys:
            outside += bdry.dist_to_bdry(Xbdry) < 0
        
        if any(outside):
            Xbdry[outside,:] = self.make_bdry_pts(torch.sum(outside), boundingbox, is_unsteady, other_bdrys)

        return Xbdry
    
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain
        distance = torch.sum( self.normal.to(X.device)*X[:,:3], dim=1) + self.constant
        
        return distance

# Periodic Boundaries
class bdry_periodic:

    def __init__(self, variable, base, top):
        
        if variable == 'x':
            self.index = 0
        if variable == 'y':
            self.index = 1
        if variable == 'z':
            self.index = 2

        self.base = base
        self.top = top

### DataLoader

class Walker_Data(torch.utils.data.Dataset):
    
    def __init__(self, num_walkers, boundingbox, boundaries):
        
        Xold = generate_interior_points(num_walkers, boundingbox, boundaries)
        #region = close_to_region(Xold)
        
        self.location = Xold
        self.num_pts = num_walkers
        #self.region = region
        
    def __len__(self):
        ### How many data points are there?
        return self.num_pts
    
    def __getitem__(self, index):
        ### Gets one sample of data
        ### 
        return self.location[index,:], index
        
class Boundary_Data(torch.utils.data.Dataset):
    
    def __init__(self, num_bdry, boundingbox, boundaries, is_unsteady):
        
        Xbdry, Ubdry = generate_boundary_points(num_bdry, boundingbox, boundaries, is_unsteady)
        
        self.location = Xbdry
        self.num_pts = num_bdry
        self.value = Ubdry
        
    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, index):
        return self.location[index,:], self.value[index,:]    
    
class Initial_Data(torch.utils.data.Dataset):
    
    def __init__(self, num_init, boundingbox, boundaries, init_con):
        
        Xinit = generate_interior_points(num_init, boundingbox, boundaries)
        Xinit[:,-1] = 0
        
        self.location = Xinit
        self.num_pts = num_init
        self.value = init_con(self.location)
        
    def __len__(self):
        ### How many data points are there?
        return self.num_pts
    
    def __getitem__(self, index):
        ### Gets one sample of data
        ### 
        return self.location[index,:], self.value[index,:]

### Functions

def generate_interior_points(num_walkers, boundingbox, boundaries):
    ### Generate points inside the domain

    X = torch.empty( (num_walkers, len(boundingbox)) )

    for ii in range(len(boundingbox)):
        X[:,ii] = (boundingbox[ii][1] - boundingbox[ii][0])*torch.rand( (num_walkers) ) + boundingbox[ii][0]

    outside = torch.zeros( X.size(0), dtype=torch.bool)
    for bdry in boundaries:
        outside += bdry.dist_to_bdry(X) < 0
    
    if any(outside):
        X[outside,:] = generate_interior_points(torch.sum(outside), boundingbox, boundaries)
        
    return X

def generate_boundary_points(num_bdry, boundingbox, boundaries, is_unsteady):
    ### Generate points along the boundary
    
    points_per_bdry = []
    utrue_per_bdry = []
    
    # Generate num_bdry points for each boundary
    for ii in range(len(boundaries)):
        bdry = boundaries[ii]
        other_bdrys = boundaries[:ii] + boundaries[ii+1:]

        # Generate boundary points
        X_in_bdry =  bdry.make_bdry_pts(num_bdry, boundingbox, is_unsteady, other_bdrys)
        U_in_bdry = bdry.bdry_cond(X_in_bdry)

        points_per_bdry.append( X_in_bdry )
        utrue_per_bdry.append( U_in_bdry )

    Xbdry = torch.cat( points_per_bdry, dim=0)
    Ubdry_true = torch.cat( utrue_per_bdry, dim=0)

    # Sample from above boundary points
    #indices = torch.multinomial( torch.arange( len(boundaries)*num_bdry, dtype=torch.float ), num_bdry)
    
    #Xbdry = Xbdry[indices,:]
    #Ubdry_true = Ubdry_true[indices,:]
    
    return Xbdry, Ubdry_true