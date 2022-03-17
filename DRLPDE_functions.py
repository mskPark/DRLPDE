###
### The functions
###

import torch
import math
import numpy as np

#######################
####################### Functions
#######################

######
###### Universal Functions
######

def make_boundaries(my_bdry):
    ### Unpack the list of boundary descriptions and make a new list with the boundary classes
    
    boundaries = []
    
    for specs in my_bdry:
        if specs[0] == 'line':
            boundaries.append( bdry_line( specs[1], specs[2], specs[3], specs[4] ) )
        elif specs[0] == 'disk':
            boundaries.append( bdry_disk( specs[1], specs[2], specs[3], specs[4] ) )
        elif specs[0] == 'ring':
            boundaries.append( bdry_ring( specs[1], specs[2], specs[3], specs[4] ) )
    
    return boundaries
                              
def generate_interior_points(num_walkers, domain, boundaries):
    ### Generate points inside the domain
    
    X = torch.empty( (num_walkers, len(domain)) )
    
    for ii in range(len(domain)):
        X[:,ii] = (domain[ii][1] - domain[ii][0])*torch.rand( (num_walkers) ) + domain[ii][0]
        
    outside = torch.zeros( X.size(0), dtype=torch.bool)
    for bdry in boundaries:
        outside += bdry.dist_to_bdry(X) > 0
    
    if any(outside):
        X[outside,:] = generate_interior_points(torch.sum(outside), domain, boundaries)
        
    return X

def generate_boundary_points(num_bdry, domain, boundaries, is_unsteady):
    ### Generate points along the boundary
    
    points_per_bdry = []
    utrue_per_bdry = []
    
    # Generate num_bdry points for each boundary
    for bdry in boundaries:
        X_in_bdry, U_in_bdry =  bdry.generate_boundary(num_bdry, domain, is_unsteady)
        points_per_bdry.append( X_in_bdry )
        utrue_per_bdry.append( U_in_bdry )
    
    Xbdry = torch.cat( points_per_bdry, dim=0)
    Ubdry = torch.cat( utrue_per_bdry, dim=0)
    
    # Sample from above boundary points
    indices = torch.multinomial( torch.linspace( 0, len(boundaries)*num_bdry - 1, len(boundaries)*num_bdry ), num_bdry)
    
    Xbdry = Xbdry[indices,:]
    Ubdry = Ubdry[indices,:]
    
    return Xbdry, Ubdry

######
###### Problem dependent functions
######


### Move Walkers
def move_Walkers_NS_steady(X, model, boundaries, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    ### Move walkers
    
    Uold = model(X)
    
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1) - dt*Uold.detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt
    
    Xnew, outside = exit_condition_steady(X.repeat(num_ghost,1), Xnew, boundaries, tol)
    
    return Xnew, Uold, outside[:num_batch]

def move_Walkers_NS_unsteady(X, model, boundaries, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    ### Move walkers
    
    Uold = model(X)
    
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1)  + torch.cat( (-dt*Uold.detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt, 
                                                -dt*torch.ones((num_batch*num_ghost,1), device=X.device, requires_grad=True)), dim=1)
    
    Xnew, outside = exit_condition_unsteady(X.repeat(num_ghost,1), Xnew, boundaries, tol)
    
    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Elliptic(X, model, boundaries, x_dim, mu, dt, num_batch, num_ghost, tol, drift, **move_walkers_param):
    ### Move walkers
    
    Uold = model(X)
    
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1) - dt*drift(X).detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt
    
    Xnew, outside = exit_condition_steady(X.repeat(num_ghost,1), Xnew, boundaries, tol)
    
    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Parabolic(X, model, boundaries, x_dim, mu, dt, num_batch, num_ghost, tol, drift, **move_walkers_param):
    ### Move walkers
    
    Uold = model(X)
    
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)

    Xnew = X.repeat(num_ghost,1)  + torch.cat( (-dt*drift(X).detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt, 
                                                -dt*torch.ones((num_batch*num_ghost,1), device=X.device, requires_grad=True)), dim=1)
    
    Xnew, outside = exit_condition_unsteady(X.repeat(num_ghost,1), Xnew, boundaries, tol)
    
    return Xnew, Uold, outside[:num_batch]


### Evaluate Model

def evaluate_model_NS(Xold, Xnew, model, dt, forcing, **eval_model_param):
    ### Evaluate the model
    
    Unew = model(Xnew) + forcing(Xnew)*dt

    return Unew
    
def evaluate_model_PDE(Xold, Xnew, model, dt, forcing, reaction, **eval_model_param):
    ### Evaluate the model, scaling with the reaction term
    
    Unew = model(Xnew)*torch.exp( reaction(Xnew)*dt) + forcing(Xnew)*dt
    
    return Unew

### Part of move_Walkers functions

def exit_condition_steady(Xold, Xnew, boundaries, tol):
    ### Calculate exit conditions
    outside = torch.zeros( Xnew.size(0), dtype=torch.bool, device=Xnew.device)
    
    for bdry in boundaries:
        outside_bdry = bdry.dist_to_bdry(Xnew) > 0
        if torch.any(outside_bdry) > 0:
            ### Bisection to get close to exit location
            ### TODO: should we take a point on the boundary (by projecting or something)
            
            Xnew[outside_bdry,:] = find_bdry_exit(Xold[outside_bdry,:], Xnew[outside_bdry,:], bdry, tol)

        outside += outside_bdry

    return Xnew, outside
    
def exit_condition_unsteady(Xold, Xnew, boundaries, tol):
    ### Calculate exit conditions
    outside = torch.zeros( Xnew.size(0), dtype=torch.bool, device=Xnew.device)
    
    for bdry in boundaries:
        outside_bdry = bdry.dist_to_bdry(Xnew) > 0
        if torch.any(outside_bdry) > 0:
            ### Bisection to get close to exit location
            ### TODO: should we take a point on the boundary (by projecting or something)
            
            Xnew[outside_bdry,:] = find_bdry_exit(Xold[outside_bdry,:], Xnew[outside_bdry,:], bdry, tol)

        outside += outside_bdry
    
    ### Check for time = 0
    ### Note: This prioritizes time exit over bdry exit
    hit_initial = Xnew[:,-1] < 0
    Xnew[hit_initial,:] = find_time_exit(Xold[hit_initial,:], Xnew[hit_initial,:], tol)

    outside += hit_initial
    
    return Xnew, outside
    
def find_bdry_exit(Xold, Xnew, bdry, tol):
    ### Bisection algorithm to find the exit between Xnew and Xold up to a tolerance 
    
    Xmid = (Xnew + Xold)/2
    
    dist = bdry.dist_to_bdry(Xmid)
    
    above_tol = dist > tol
    below_tol = dist < -tol
    
    if torch.sum(above_tol + below_tol) > 0:
        Xnew[above_tol,:] = Xmid[above_tol,:]
        Xold[below_tol,:] = Xmid[below_tol,:]
        
        Xmid[above_tol + below_tol,:] = find_bdry_exit(Xold[above_tol + below_tol,:], Xnew[above_tol + below_tol,:], bdry, tol)

    return Xmid
            
def find_time_exit(Xold, Xnew, tol):
    ### Bisection algorithm to find the time exit up to a tolerance
    Xmid = (Xnew + Xold)/2
    above_tol = Xmid[:,-1] > tol
    below_tol = Xmid[:,-1] < tol
    if torch.sum(above_tol + below_tol) > 0:
        Xnew[above_tol,:] = Xmid[above_tol,:]
        Xold[below_tol,:] = Xmid[below_tol,:]
        
        Xmid = find_time_exit(Xold[above_tol + below_tol,:], Xnew[above_tol + below_tol,:], tol)
    
    # Project the time to the initial time?
    #Xmid[:,-1] = time_range[0]
    
    return Xmid

##################      
##################      Class Structures
##################

################## 2D Boundaries #######################

class bdry_disk:
    ### Class structure for a 2D solid disk boundary, the domain being outside the disk
    
    bdry_type = 'disk'
    
    def __init__(self, centre, radius, endpoints, bdry_cond):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius

        self.bdry_cond = bdry_cond
        
        ### Convert endpoints to angles
        ###
        ### Use angles to sample along disk
        if len(endpoints) == 0:
            self.angles = [0, 2*math.pi]
        else:
            self.angles = []
            for point in endpoints:
                angles.append( np.angle( np.complex( point[0] - self.centre[0], point[1] - self.centre[1] ) ) )
            if angles[1] < angles[0]:
                angles[1] += 2*math.pi
            self.angles = angles
            
            
    def generate_boundary(self, num_bdry, domain, is_unsteady):
        theta = (self.angles[1] - self.angles[0])*torch.rand((num_bdry)) + self.angles[0]
        
        if is_unsteady:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(theta) + self.centre[1],
                                (domain[-1][1] - domain[-1][0])*torch.rand((num_bdry)) + domain[-1][0]),dim=1 )  
        else:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )
        
        Utrue = self.bdry_cond(Xbdry)
        
        return Xbdry, Utrue
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = outside domain
        ### negative = inside domain
        distance = ( self.radius - torch.norm( X[:,:2] - self.centre.to(X.device),dim=1) )
        return distance
    
    #def closest_point(self, X):
        ### Find the closest point on the boundary by making the radius
        #Xnew = X/torch.norm( X[:,:2] - self.centre.to(X.device), dim=1)
        #return Xnew
    
    def plot_bdry(self, num_bdry):
        theta = torch.linspace(self.angles[0], self.angles[1], num_bdry)
        Xplot = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )
        
        return Xplot
class bdry_ring:
    ### Class structure for a circle boundary, the inside being the domain
    bdry_type = 'ring'
    
    def __init__(self, centre, radius, endpoints, bdry_cond):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius

        self.bdry_cond = bdry_cond
        
        ### Convert endpoints to angles
        ###
        ### Use angles to sample along disk
        if len(endpoints) == 0:
            self.angles = [0, 2*math.pi]
        else:
            self.angles = []
            for point in endpoints:
                angles.append( np.angle( np.complex( point[0] - self.centre[0], point[1] - self.centre[1] ) ) )
            if angles[1] < angles[0]:
                angles[1] += 2*math.pi
            self.angles = angles
            
            
    def generate_boundary(self, num_bdry, domain, is_unsteady):
        theta = (self.angles[1] - self.angles[0])*torch.rand((num_bdry)) + self.angles[0]
        
        if is_unsteady:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(theta) + self.centre[1],
                                (domain[-1][1] - domain[-1][0])*torch.rand((num_bdry)) + domain[-1][0]),dim=1 )  
        else:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )
            
        Utrue = self.bdry_cond(Xbdry)
        
        return Xbdry, Utrue
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = outside domain
        ### negative = inside domain
        distance = ( torch.norm( X[:,:2] - self.centre.to(X.device),dim=1) - self.radius )
        return distance
    
    #def closest_point(self, X):
        ### Find the closest point on the boundary by making the radius
        #Xnew = X/torch.norm( X[:,:2] - self.centre.to(X.device), dim=1)
        #return Xnew
    
    def plot_bdry(self, num_bdry):
        theta = torch.linspace(self.angles[0], self.angles[1], num_bdry)
        Xplot = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )
        
        return Xplot
    
    
class bdry_line:
    ### Class structure for a line boundary
    ###       normal vector points outside
    
    bdry_type = 'line'
    
    def __init__(self, point, normal, endpoints, bdry_cond):
        self.point = torch.tensor(  point )
        self.normal = torch.tensor( normal )
        self.constant = -sum( self.normal*self.point )
        
        self.bdry_cond = bdry_cond
        
        self.endpoints = torch.tensor(endpoints)
        
    def generate_boundary(self, num_bdry, domain, is_unsteady):
        
        if is_unsteady:
            Xbdry = torch.cat( ( (self.endpoints[1] - self.endpoints[0] )*torch.rand((num_bdry,1)) + self.endpoints[0],
                                 (domain[-1][1] - domain[-1][0])*torch.rand((num_bdry,1)) + domain[-1][0]), dim=1)
        else:    
            Xbdry = ( self.endpoints[1] - self.endpoints[0] )*torch.rand((num_bdry,1)) + self.endpoints[0]
 
        Utrue = self.bdry_cond(Xbdry)

        return Xbdry, Utrue
    
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = outside domain
        ### negative = inside domain
        distance = torch.sum( self.normal.to(X.device)*X[:,:2], dim=1) + self.constant
        
        return distance
    
    #def closest_point(self, X):
        ### Find the closest point given an (x,y) by going in the normal direction
        #distance = dist_to_bdry(X)
        #X[:,:2] = X[:,:2] + distance[:,None]*self.normal.to(X.device)
        
        #return X
    
    def plot_bdry(self, num_bdry):
        Xplot = ( self.endpoints[1] - self.endpoints[0] )*torch.linspace(0, 1, num_bdry)[:,None] + self.endpoints[0]
        
        return Xplot

##################  Dataloader   #######################
    
class Walker_Data(torch.utils.data.Dataset):
    
    def __init__(self, num_walkers, domain, boundaries):
        
        Xold = generate_interior_points(num_walkers, domain, boundaries)
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
    
    def __init__(self, num_bdry, domain, boundaries, is_unsteady):
        
        Xbdry, Ubdry = generate_boundary_points(num_bdry, domain, boundaries, is_unsteady)
        
        self.location = Xbdry
        self.num_pts = num_bdry
        self.value = Ubdry
        
    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, index):
        return self.location[index,:], self.value[index,:]    
    
class Initial_Data(torch.utils.data.Dataset):
    
    def __init__(self, num_init, domain, boundaries, init_con):
        
        Xinit = generate_interior_points(num_init, domain, boundaries)
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