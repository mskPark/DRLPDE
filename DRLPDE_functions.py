import torch
import math
import numpy as np

################################
##############################
###########################

### Class structures for problem parameters

### Domain parameters
class Domain:

    def __init__(self, is_unsteady, boundingbox, list_of_bdry, any_periodic, periodic_bdry):
        
        self.boundingbox = boundingbox
        self.is_unsteady = is_unsteady

        self.num_of_boundaries = len(list_of_bdry)
        
        # Unpack boundaries descriptions
        self.boundaries = []
        for specs in list_of_bdry:
            ### 2D boundaries
            if specs['type'] == 'line':
                self.boundaries.append( bdry_line( point = specs['point'], 
                                                   normal = specs['normal'],
                                                   endpoints = specs['endpoints'],
                                                   boundary_condition = specs['boundary_condition'] ))
            
            if specs['type'] == 'disk':
                self.boundaries.append( bdry_disk( centre = specs['centre'],
                                                   radius = specs['radius'], 
                                                   boundary_condition = specs['boundary_condition'] ))
            
            if specs['type'] == 'ring':
                self.boundaries.append( bdry_ring( centre = specs['centre'],
                                                   radius = specs['radius'], 
                                                   boundar_condition = specs['boundary_condition'] ))

            if specs['type'] == 'ball':
                self.boundaries.append( bdry_ball( centre = specs['centre'],
                                                   radius = specs['radius'], 
                                                   boundary_condition = specs['boundary_condition'] ))
            
            ### 3D boundaries - Note: 2D and 3D boundaries not compatible with each other
            if specs['type'] == 'sphere':
                self.boundaries.append( bdry_sphere( centre = specs['centre'], 
                                                     radius = specs['radius'], 
                                                     boundary_condition = specs['boundary_condition']))

            if specs['type'] == 'cylinder':
                self.boundaries.append( bdry_cylinder( centre = specs['centre'],
                                                       radius = specs['radius'],
                                                       ### TODO axis of rotation
                                                       boundary_condition = specs['boundary_condition'] ))
            if specs['type'] == 'plane':
                self.boundaries.append( )

            if specs[0] == 'line':
                self.boundaries.append( bdry_line( specs[1], specs[2], specs[3], specs[4] ) )
            elif specs[0] == 'disk':
                self.boundaries.append( bdry_disk( specs[1], specs[2], specs[3], specs[4] ) )
            elif specs[0] == 'ring':
                self.boundaries.append( bdry_ring( specs[1], specs[2], specs[3], specs[4] ) )
            elif specs[0] == 'ball':
                self.boundaries.append( bdry_ball( specs[1], specs[2], specs[3] ) )
            elif specs[0] == 'sphere':
                self.boundaries.append( bdry_sphere( specs[1], specs[2], specs[3] ) )
            elif specs[0] == 'cylinder': ### Add axis to cylinder
                self.boundaries.append( bdry_cylinder( specs[1], specs[2], specs[3] ) )
            elif specs[0] == 'plane':
                self.boundaries.append( bdry_plane( specs[1], specs[2], specs[3], specs[4] ) )
        
        # Unpack any periodic boundaries
        self.any_periodic=any_periodic
        if any_periodic:
            self.periodic_index = periodic_bdry[0]
            self.periodic_base = periodic_bdry[1]
            self.periodic_top = periodic_bdry[2]

#######################
####################### Functions
#######################

######
###### Universal Functions
######
                              
def generate_interior_points(num_walkers, Domain):
    ### Generate points inside the domain
    
    X = torch.empty( (num_walkers, len(Domain.boundingbox)) )
    
    for ii in range(len(boundingbox)):
        X[:,ii] = (boundingbox[ii][1] - boundingbox[ii][0])*torch.rand( (num_walkers) ) + boundingbox[ii][0]
        
    outside = torch.zeros( X.size(0), dtype=torch.bool)
    for bdry in boundaries.list_bdry:
        outside += bdry.dist_to_bdry(X) > 0
    
    if any(outside):
        X[outside,:] = generate_interior_points(torch.sum(outside), boundingbox, boundaries)
        
    return X

def generate_boundary_points(num_bdry, Domain):
    ### Generate points along the boundary
    
    points_per_bdry = []
    utrue_per_bdry = []
    
    # Generate num_bdry points for each boundary
    for bdry in boundaries.list_bdry:
        X_in_bdry, U_in_bdry =  bdry.generate_boundary(num_bdry, boundingbox, is_unsteady)
        points_per_bdry.append( X_in_bdry )
        utrue_per_bdry.append( U_in_bdry )
    
    Xbdry = torch.cat( points_per_bdry, dim=0)
    Ubdry_true = torch.cat( utrue_per_bdry, dim=0)
    
    # Sample from above boundary points
    indices = torch.multinomial( torch.linspace( 0, boundaries.how_many*num_bdry - 1, boundaries.how_many*num_bdry ), num_bdry)
    
    Xbdry = Xbdry[indices,:]
    Ubdry_true = Ubdry_true[indices,:]
    
    return Xbdry, Ubdry_true

######
###### Problem dependent functions
######


### Move Walkers
def move_Walkers_NS_steady(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    ### Move walkers
    
    Uold = model(X)
    
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1) - dt*Uold.detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt
    
    Xnew, outside = exit_condition_steady(X.repeat(num_ghost,1), Xnew, boundaries, tol)
    
    return Xnew, Uold, outside[:num_batch]

def move_Walkers_NS_unsteady(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    ### Move walkers
    
    Uold = model(X)
    
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1)  + torch.cat( (-dt*Uold.detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt, 
                                                -dt*torch.ones((num_batch*num_ghost,1), device=X.device, requires_grad=True)), dim=1)
    
    Xnew, outside = exit_condition_unsteady(X.repeat(num_ghost,1), Xnew, boundaries, tol)
    
    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Stokes_steady(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    ### Move walkers
    
    Uold = model(X)
    
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1) + np.sqrt(2*mu)*Zt
    
    Xnew, outside = exit_condition_steady(X.repeat(num_ghost,1), Xnew, boundaries, tol)
    
    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Stokes_unsteady(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    ### Move walkers
    
    Uold = model(X)
    
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1)  + torch.cat( ( np.sqrt(2*mu)*Zt, 
                                                -dt*torch.ones((num_batch*num_ghost,1), device=X.device, requires_grad=True)), dim=1)
    
    Xnew, outside = exit_condition_unsteady(X.repeat(num_ghost,1), Xnew, boundaries, tol)
    
    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Elliptic(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, drift, **move_walkers_param):
    ### Move walkers
    
    Uold = model(X)
    
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1) - dt*drift(X).detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt
    
    Xnew, outside = exit_condition_steady(X.repeat(num_ghost,1), Xnew, boundaries, tol)
    
    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Parabolic(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, drift, **move_walkers_param):
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
            ### Question: 
            ###     Should we take a point on the boundary (by projecting or something)
            ###     or is within tol good enough?
            Xnew[outside_bdry,:] = find_bdry_exit(Xold[outside_bdry,:], Xnew[outside_bdry,:], bdry, tol)

        outside += outside_bdry
    
    ### Check for time = 0
    ### Note: This prioritizes time exit over bdry exit
    ### Question: 
    ###     Should we take a point at the initial time (by projecting or something)
    ###     or is within tol good enough?
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
    below_tol = Xmid[:,-1] < -tol

    if torch.sum(above_tol + below_tol) > 0:
        Xnew[below_tol,:] = Xmid[below_tol,:]
        Xold[above_tol,:] = Xmid[above_tol,:]
        
        Xmid[above_tol + below_tol,:] = find_time_exit(Xold[above_tol + below_tol,:], Xnew[above_tol + below_tol,:], tol)

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
            
            
    def generate_boundary(self, num_bdry, boundingbox, is_unsteady):
        theta = (self.angles[1] - self.angles[0])*torch.rand((num_bdry)) + self.angles[0]
        
        if is_unsteady:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(theta) + self.centre[1],
                                (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]),dim=1 )  
        else:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )
        
        Utrue = self.bdry_cond(Xbdry)
        
        return Xbdry, Utrue
            
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
            
            
    def generate_boundary(self, num_bdry, boundingbox, is_unsteady):
        theta = (self.angles[1] - self.angles[0])*torch.rand((num_bdry)) + self.angles[0]
        
        if is_unsteady:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(theta) + self.centre[1],
                                (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]),dim=1 )  
        else:
            Xbdry = torch.stack((self.radius*torch.cos(theta) + self.centre[0],
                             self.radius*torch.sin(theta) + self.centre[1]),dim=1 )
            
        Utrue = self.bdry_cond(Xbdry)
        
        return Xbdry, Utrue
            
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
    
    bdry_type = 'line'
    
    def __init__(self, point, normal, endpoints, bdry_cond):
        self.point = torch.tensor(  point )
        self.normal = torch.tensor( normal )
        self.constant = -sum( self.normal*self.point )
        
        self.bdry_cond = bdry_cond
        
        self.endpoints = torch.tensor(endpoints)
        
    def generate_boundary(self, num_bdry, boundingbox, is_unsteady):
        
        if is_unsteady:
            Xbdry = torch.cat( ( (self.endpoints[1] - self.endpoints[0] )*torch.rand((num_bdry,1)) + self.endpoints[0],
                                 (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry,1)) + boundingbox[-1][0]), dim=1)
        else:    
            Xbdry = ( self.endpoints[1] - self.endpoints[0] )*torch.rand((num_bdry,1)) + self.endpoints[0]
 
        Utrue = self.bdry_cond(Xbdry)

        return Xbdry, Utrue
    
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain
        distance = torch.sum( self.normal.to(X.device)*X[:,:2], dim=1) + self.constant
        
        return distance
    
    def plot_bdry(self, num_bdry):
        Xplot = ( self.endpoints[1] - self.endpoints[0] )*torch.linspace(0, 1, num_bdry)[:,None] + self.endpoints[0]
        
        return Xplot

################## 3D Boundaries #######################

class bdry_ball:
    ### Class structure for a 3D solid ball boundary, the domain being outside the ball
    
    bdry_type = 'ball'
    
    def __init__(self, centre, radius, bdry_cond):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius

        self.bdry_cond = bdry_cond
        
            
    def generate_boundary(self, num_bdry, boundingbox, is_unsteady):
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
        Utrue = self.bdry_cond(Xbdry)
        
        return Xbdry, Utrue
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        distance = ( torch.norm( X[:,:3] - self.centre.to(X.device),dim=1) - self.radius )
        return distance
    
class bdry_sphere:
    ### Class structure for a 3D hollow sphere boundary, the domain being inside the sphere
    
    bdry_type = 'sphere'
    
    def __init__(self, centre, radius, bdry_cond):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius

        self.bdry_cond = bdry_cond
        
            
    def generate_boundary(self, num_bdry, boundingbox, is_unsteady):
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
        Utrue = self.bdry_cond(Xbdry)
        
        return Xbdry, Utrue
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain

        distance = ( self.radius - torch.norm( X[:,:3] - self.centre.to(X.device),dim=1) )
        return distance
    
class bdry_cylinder:
    ### Class structure for inside a cylindrical shell
    ### Centre: One end of the cylinder
    ### Points in z-direction =

    ### todo include axis: Points in direction, length of axis determines how long
    
    bdry_type = 'cylinder'
    
    def __init__(self, centre, radius, axis, bdry_cond):
        ### Centre and Radius
        self.centre = torch.tensor( centre )
        self.radius = radius
        #self.axis = torch.tensor( axis )

        self.bdry_cond = bdry_cond
        
            
    def generate_boundary(self, num_bdry, boundingbox, is_unsteady):
        ### Cylindrical coordinates
        ###
        ### x = radius*cos(theta)
        ### y = radius*sin(theta)
        ### z = z

        theta = 2*math.pi*torch.rand( (num_bdry))
        
        if is_unsteady:
            Xbdry = torch.stack((self.radius*torch.sin(phi)*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(phi)*torch.sin(theta) + self.centre[1],
                                (boundingbox[-2][1] - boundingbox[-2][0])*torch.rand((num_bdry)) + boundingbox[-2][0],
                                (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]),dim=1 )  
        else:
            Xbdry = torch.stack((self.radius*torch.sin(phi)*torch.cos(theta) + self.centre[0],
                                 self.radius*torch.sin(phi)*torch.sin(theta) + self.centre[1],
                                 (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry)) + boundingbox[-1][0]), dim=1 )  
        Utrue = self.bdry_cond(Xbdry)
        
        return Xbdry, Utrue
            
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = outside domain
        ### negative = inside domain
        distance = ( self.radius - torch.norm( X[:,:2] - self.centre.to(X.device),dim=1) )
        return distance

class bdry_plane:
    ### Class structure for a plane in 3D space
    ### normal vector points inside
    ### corners should be opposite

    bdry_type = 'plane'
    
    def __init__(self, point, normal, corners, bdry_cond):
        self.point = torch.tensor(  point )
        self.normal = torch.tensor( normal )
        self.constant = -sum( self.normal*self.point )
        
        self.corners = torch.tensor( corners )
        
        self.bdry_cond = bdry_cond
        
    def generate_boundary(self, num_bdry, boundingbox, is_unsteady):
        
        if is_unsteady:
            Xbdry = torch.cat( ( (self.endpoints[1] - self.endpoints[0] )*torch.rand((num_bdry,1)) + self.endpoints[0],
                                 (boundingbox[-1][1] - boundingbox[-1][0])*torch.rand((num_bdry,1)) + boundingbox[-1][0]), dim=1)
        else:    
            Xbdry = ( self.corners[1] - self.corners[0] )*torch.rand((num_bdry,1)) + self.corners[0]
 
        Utrue = self.bdry_cond(Xbdry)

        return Xbdry, Utrue
    
    def dist_to_bdry(self, X):
        ### Signed distance to boundary
        ### positive = inside domain
        ### negative = outside domain
        distance = torch.sum( self.normal.to(X.device)*X[:,:3], dim=1) + self.constant
        
        return distance

##################               #######################
##################  Dataloader   #######################
##################               #######################

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