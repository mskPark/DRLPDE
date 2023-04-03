import torch
import math
import numpy as np

# 2-Dimensional Walls

class ball:
    ### Class structure for a spherical boundary, the domain being outside the ball
    
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

class sphere:
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
    
class cylinder:
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

class plane:
    
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

class funtype1:
    ### (x, y, f(x,y))
    pass

class funtype2:
    ### (x, g(x,z), z)
    pass

class funtype3:
    ### (h(y,z), y, z)
    pass

# 3-Dimensional Solid Walls

class solidball:
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
    