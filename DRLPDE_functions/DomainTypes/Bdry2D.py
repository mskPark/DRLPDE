###
### Boundary Classes for 2 dimensional domains
###

### Initialization: All attributes that distinguish them
### Methods:
###   unit_normal(self, bdry_pt) returns a normal vector at a point on the boundary
###            !!NORMAL VECTOR POINTS INWARDS!!
###
###   make_bdry_pts(self) returns a uniform/random sampling of the boundary
###   
###   dist_to_bdry(self, dom_pt) returns the signed distance to the boundary
###
###   

import torch
import math
import numpy as np

### Class structure for a 2D solid disk boundary, the domain being outside the disk
class bdry_disk:

    def __init__(self, centre, radius):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius

        ### Area of the boundary
        self.area = math.pi*self.radius**2

    def unit_normal(self, bdry_pt):
        ### Input: bdry_pt
        ### Output the unit normal vector at that point

        return (bdry_pt- self.centre)/self.radius
            
    def make_bdry_pts(self, num_bdry, boundingbox, is_unsteady):
        ### Make points along the boundary as well as the interior
        ### Number of points in the interior should be the fraction of the domain area

        # Random 
        #theta = (self.angles[1] - self.angles[0])*torch.rand((2*num_bdry)) + self.angles[0]
        #inside_rad = self.radius*torch.sqrt( torch.rand(num_bdry) )

        # Temporary uniform meshgrid
        #theta_r = torch.cartesian_prod( torch.linspace(0, 2*math.pi, 2**7), self.radius*torch.sqrt( torch.linspace(0,1,2**4) ))
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

### Class structure for a circle boundary, the inside being the domain
class bdry_ring:
    
    def __init__(self, centre, radius):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius

    def unit_normal(self, bdry_pt):

        pass
            
            
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

    def unit_normal(self, bdry_pt):

        pass

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


class bdry_wedge:

    def __init__(self, centre, radius, endpoints, boundary_condition):
### Use angles to sample along disk
        if len(endpoints) == 0:
            self.angles = [0, 2*math.pi]
        else:
            self.angles = []
            for point in endpoints:
                self.angles.append( np.angle( np.complex( point[0] - self.centre[0], point[1] - self.centre[1] ) ) )
            if self.angles[1] < self.angles[0]:
                self.angles[1] += 2*math.pi