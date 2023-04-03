import torch
import math
import numpy as np

# 1-Dimensional Walls

# Removed: Checking other walls to see if outside
# Walls must be non-overlapping

    ### Check if outside other bdrys
    ### and remake bdry points

    #outside = torch.zeros(Xwall.size(0), dtype=torch.bool)

    #for bdry in other_bdrys:
    #    outside += bdry.distance(Xwall) < 0
    
    #if any(outside):
    #    Xwall[outside,:] = self.make_points(torch.sum(outside), other_bdrys)


class circle:
    ### Class structure for a circle boundary, the outside being the domain
    def __init__(self, centre, radius, bc):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius
        self.measure = 2*math.pi*radius
        self.dim = 1
        self.bc = bc
            
    def make_points(self, num):
        
        theta = 2*math.pi*torch.rand(num)

        Xwall = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )

        return Xwall
            
    def distance(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        distance = (torch.norm( X[:,:2] - self.centre.to(X.device),dim=1) - self.radius)

        return distance
    
    def plot(self, num_bdry):
        pass

class ring:
    ### Class structure for a circle boundary, the inside being the domain
    
    def __init__(self, centre, radius, bc):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius
        self.measure = 2.0*math.pi*radius
        self.dim = 1
        self.bc = bc
            
    def make_points(self, num):
        
        theta = 2*math.pi*torch.rand(num)

        Xwall = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )
        return Xwall
            
    def distance(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        distance = (self.radius - torch.norm( X[:,:2] - self.centre.to(X.device),dim=1))

        return distance
    
    def plot(self):
        pass
       
class line:
    ### Class structure for a line boundary
    ###       normal vector points inside
    
    def __init__(self, endpoints, normal, bc):
        self.endpoints = torch.tensor(endpoints)
        self.normal = torch.tensor( normal )
        self.constant = -sum( self.normal*self.endpoints[0] )
        self.measure = torch.norm( self.endpoints[0] - self.endpoints[1] )
        self.dim = 1
        self.bc = bc
        
    def make_points(self, num):
        
        Xwall = ( self.endpoints[1] - self.endpoints[0] )*torch.rand(num, 1) + self.endpoints[0]
        
        return Xwall
    
    def distance(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain
        distance = torch.sum( self.normal.to(X.device)*X[:,:2], dim=1) + self.constant
        
        return distance
    
    def plot_bdry(self, num_bdry):
        Xplot = ( self.endpoints[1] - self.endpoints[0] )*torch.linspace(0, 1, num_bdry)[:,None] + self.endpoints[0]
        
        return Xplot

class polar:

    ### Class structure for a polar curve, the inside being the domain
    def __init__(self, polar_eq, bc):
        ### Centre and Radius
        self.polar_eq = polar_eq

        self.dim = 1
        self.bc = bc

        ### TODO make general
        self.measure = 75.4194
            
    def make_points(self, num):
        
        theta = 2*math.pi*torch.rand(num)
        r = self.polar_eq(theta)

        Xwall = torch.stack((r*torch.cos(theta),
                             r*torch.sin(theta)),dim=1 )

        return Xwall
            
    def distance(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        theta = torch.atan2( X[:,1], X[:,0])

        distance = self.polar_eq(theta) - torch.norm( X[:,:2] ,dim=1)

        return distance
    
    def plot(self, num_bdry):
        pass

class box:
    ### Class structure for a box mesh, the inside is the domain

    def __init__(self, xint, yint):
        ### Centre and Radius
        self.xint = torch.tensor(xint)
        self.yint = torch.tensor(yint)

        self.measure = 2*( xint[1] - xint[0] ) + 2*( yint[1] - yint[0] ) 
        self.dim = 1
        
    def rhs(self, num, model):
        ### Make num points along each wall

        numpts = num-2

        dx = (self.xint[1] - self.xint[0])/(num -1)
        dy = (self.yint[1] - self.yint[0])/(num -1)

        ratio = ((dy/dx)**2)

        Xlin = torch.linspace( self.xint[0], self.xint[1], num)[1:-1]
        Ylin = torch.linspace( self.yint[0], self.yint[1], num)[1:-1]

        Xleft = torch.stack( (self.xint[0].repeat(numpts), Ylin), dim=1 ).requires_grad_(True)
        Xright = torch.stack( (self.xint[1].repeat(numpts), Ylin), dim=1 ).requires_grad_(True)
        Xbot = torch.stack( (Xlin, self.yint[0].repeat(numpts)), dim=1 ).requires_grad_(True) 
        Xtop = torch.stack( (Xlin, self.yint[0].repeat(numpts)), dim=1 ).requires_grad_(True) 

        Xwall = torch.cat( (Xleft, Xbot, Xtop, Xright), dim=0)

        b = torch.zeros((num-2)**2,1)

        b[:numpts] += -model(Xleft)*ratio
        b[np.arange(0,numpts**2, numpts)] += -model(Xbot)
        b[np.arange(numpts-1, numpts**2, numpts)] += -model(Xtop)
        b[-numpts:] += -model(Xright)*ratio
        
        return b, Xwall
    
    def make_laplace(self, num):

        # Maybe diag makes dtype float64 by default

        numpts = num-2

        dx = (self.xint[1] - self.xint[0])/(num -1)
        dy = (self.yint[1] - self.yint[0])/(num -1)

        ratio = ((dy/dx)**2)

        diag = torch.diag(-2.0*(1 + ratio).repeat(numpts**2))
        offdiag = torch.diag( torch.tensor(1.0).repeat(numpts**2-1), diagonal = 1 ) + torch.diag(  torch.tensor(1.0).repeat(numpts**2-1), diagonal = -1 )
        offdiag[np.arange(numpts,numpts**2, numpts), np.arange(numpts-1,numpts**2-1, numpts)] = 0.0 
        offdiag[np.arange(numpts-1,numpts**2-1, numpts), np.arange(numpts,numpts**2, numpts)] = 0.0 

        offoffdiag = torch.diag(ratio.repeat(numpts**2 - numpts), diagonal=numpts ) + torch.diag(ratio.repeat(numpts**2 - numpts), diagonal=-numpts )

        L = diag + offdiag + offoffdiag

        return L
    
    def distance(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        distance = torch.max( torch.stack( [( self.xint[0] - X[:,0] ),
                                            ( X[:,0] - self.xint[1] ),
                                            ( self.yint[0] - X[:,1] ),
                                            ( X[:,1] - self.yint[1] ) ], dim=1 ), dim=1)[0]

        return distance
    
    def plot(self, num_bdry):
        pass

class wedge:
    ### Use angles to sample along disk

    #if len(endpoints) == 0:
    #    self.angles = [0, 2*math.pi]
    #else:
    #    self.angles = []
    #    for point in endpoints:
    #        self.angles.append( np.angle( np.complex( point[0] - self.centre[0], point[1] - self.centre[1] ) ) )
    #    if self.angles[1] < self.angles[0]:
    #        self.angles[1] += 2*math.pi

    pass

class type1:
    ### (x, g(x)), range over x
    ### Need to know whether its bottom or top
    pass

class type2:
    ### (h(y), y), range over y
    ### Need to know whether its left or right
    pass

# Not needed
class inletoutlet:
    pass

# 2-Dimensional Solid Walls

class disk:
    ### Class structure for a 2D solid disk, the domain is outside the disk
    
    def __init__(self, centre, radius, bc):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius
        self.measure = math.pi*radius**2
        self.dim = 2
        self.bc = bc
            
    def make_points(self, num):
        ### Make points inside the disk

        # The disk
        r_th = torch.stack( (self.radius*torch.sqrt( torch.rand(num) ), 2*math.pi*torch.rand(num)), dim=1 )
        
        Xwall = torch.stack( (r_th[:,0]*torch.cos(r_th[:,1]) + self.centre[0],
                              r_th[:,0]*torch.sin(r_th[:,1]) + self.centre[1]), dim=1)

        return Xwall
            
    def distance(self, X):
        ### Signed distance to wall
        ### positive = inside domain
        ### negative = outside domain

        distance = ( torch.norm(X[:,:2] - self.centre.to(X.device),dim=1) - self.radius )
        return distance
    
    def plot(self, num):
        ### Give uniformly spaced points along the wall to plot
        pass
