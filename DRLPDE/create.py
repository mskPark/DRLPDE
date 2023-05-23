###
### This module contains the classes to create the domain and points
###

import torch
import math
import numpy as np
import DRLPDE.bdry as bdry

# Sum of Squared Error
SSE = torch.nn.MSELoss(reduction='none')

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

    def __init__(self, problem_parameters):

        self.boundingbox = problem_parameters['Boundaries'][0]

        list_of_walls = problem_parameters['Boundaries'][1]
        solid_walls = problem_parameters['Boundaries'][2]
        inlet_outlet = problem_parameters['Boundaries'][3]
        list_of_periodic_ends = problem_parameters['Boundaries'][4]
        mesh = problem_parameters['Boundaries'][5]

        # Dirichlet walls
        self.wall = []
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
                self.wall.append( bdry.polar( r = specs['equation'],
                                              dr = specs['derivative'],
                                              bc = specs['boundary_condition']))

            ### 3D walls - Note: 2D and 3D  walls not compatible with each other
            if specs['type'] == 'sphere':
                self.wall.append( bdry.sphere( centre = specs['centre'], 
                                                     radius = specs['radius'], 
                                                     bc = specs['boundary_condition']))
            
            if specs['type'] == 'ball':
                self.wall.append( bdry.ball( centre = specs['centre'],
                                                   radius = specs['radius'],
                                                   bc = specs['boundary_condition']))
            
            if specs['type'] == 'cylinder':
                self.wall.append( bdry.cylinder( centre = specs['centre'],
                                                       radius = specs['radius'],
                                                       ### TODO axis of rotation
                                                       bc = specs['boundary_condition'] ))
            
            if specs['type'] == 'plane':
                self.wall.append( bdry.plane( point = specs['point'],
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
        self.exitflag = self.wall + self.inletoutlet

        # For checking whether points are inside/outside the domain
        self.inside = self.wall + self.inletoutlet + self.mesh

        # Calculate volume of domain
        # Crashes for low tolerance: Need too many points
        self.volume = self.volumeDomain(1e-2)

    ### Calculate volume of the domain
    def volumeDomain(self, std):
        # Approximates volume of domain through Monte Carlo Integration
        # Sample from boundingbox B, estimate volume of domain D as
        # vol(D) = vol(B)*(fraction of samples inside D)
        # standard error = sqrt( vol(B)**2 (1-frac)*frac)/(N-1) ) ~~ volB/2sqrt(N) for (1-frac)frac maximized

        volB = 1.0

        # Calculate volume of boundingbox
        for ii in range(len(self.boundingbox)):
            volB = volB*(self.boundingbox[ii][1] - self.boundingbox[ii][0])

        # Calculate number needed to get std within tol (approximate)
        num = np.int( (volB/std/2)**2 ) 
        X = torch.empty( (num, len(self.boundingbox)) )

        # Uniformly sample from boundingbox
        for ii in range(len(self.boundingbox)):
            X[:,ii] = (self.boundingbox[ii][1] - self.boundingbox[ii][0])*torch.rand( (num) ) + self.boundingbox[ii][0]

        outside = torch.zeros( X.size(0), dtype=torch.bool)
        for wall in self.inside:
            outside += wall.distance(X) < 0
        
        frac = (num - torch.sum(outside))/num
        volD = volB*frac

        return volD

    def integrate(self, X, num, F):

        integral = self.volume*torch.sum(F)/num
        return integral

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

### Points
class thePoints:
    
    # TODO Try LBFGS
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
    # def closure():
    #     if torch.is_grad_enabled():
    #        optimizer.zero_grad()
    #     output = model(input)
    #     loss = loss_fn(output, target)
    #     if loss.requires_grad:
    #        loss.backward()
    #     return loss

    def __init__(self, num, domain, model, problem_parameters, solver_parameters, dev):
        # Organize some variables
        input_dim = problem_parameters['input_dim']
        input_range = problem_parameters['input_range']
        unsteady = bool(input_dim[1])
        var_train = problem_parameters['var_train']
        interior_target = problem_parameters['InteriorTarget']
        learningrate = solver_parameters['learningrate']

        bdry_lr = 1e-5
        Linf_lr = 0

        # Interior Points
        self.toTrain = [InteriorPoints(num, domain, input_dim, input_range, dev)]
        self.target = [interior_target]
        self.var = [var_train]
        self.integrate = [domain.integrate]
        self.L2optimizers = [torch.optim.Adam(model.parameters(), lr=learningrate)]
        self.Linfoptimizers = [torch.optim.Adam(model.parameters(), lr=Linf_lr)]
        #self.scheduler = [torch.optim.lr_scheduler.MultiStepLR(self.L2optimizers[0], milestones=[500], gamma=0.1)]
        
        self.reject = [torch.tensor([], dtype=torch.int64)]

        # domain.wall + domain.solid
        for bdry in domain.wall + domain.solid:
            nb = num_points_wall(num, bdry.measure, domain.volume, input_dim[0], bdry.dim)

            self.toTrain.append(BCPoints(nb, domain, bdry, input_dim, input_range, dev))
            self.target.append(Dirichlet_target)
            self.var.append({'true':bdry.bc})
            self.integrate.append( bdry.integrate )
            self.L2optimizers.append(torch.optim.Adam(model.parameters(), lr=bdry_lr))
            self.Linfoptimizers.append(torch.optim.Adam(model.parameters(), lr=Linf_lr))
            self.reject.append(torch.tensor([], dtype=torch.int64))

        # TODO domain.inletoutlet
        for inletoutlet in domain.inletoutlet:
            nb = num_points_wall(num, bdry.measure, domain.volume, input_dim[0] + input_dim[1], bdry.dim)
            self.toTrain.append(BCPoints(nb, inletoutlet, input_dim, input_range))
            self.target.append(Inletoutlet_target)
            self.var.append(inletoutlet.bc)
            self.L2optimizers.append()
            self.Linfoptimizers.append()
            self.reject.append([])
        # TODO mesh
        for mesh in domain.mesh:
            self.toTrain.append(MeshPoints(num_mesh, mesh))
            self.target.append()
            self.var.append()
            self.L2optimizers.append()
            self.Linfoptimizers.append()
            self.reject.append([])
        # TODO Unsteady problem
        if unsteady:
            self.toTrain.append()
            self.target.append()
            self.var.append()
            self.L2optimizers.append()
            self.Linfoptimizers.append()
            self.reject.append([])

        # How many points
        self.numtype = int(len(self.toTrain))

    def TrainL2LinfLoss(self, model, dev, numbatch, squaredlosses, importance_sampling=False):

        # dev, numbatch
        # target( X, model, volume, var_train) 
        # max_loss, importance sampling

        losses = np.zeros( (self.numtype,2) )

        for ii in range(self.numtype):
            # Organize into batch
            numpts = self.toTrain[ii].__len__()
            Batch = torch.utils.data.DataLoader(self.toTrain[ii], batch_size=numbatch, shuffle=True)

            # L2 and Linf losses
            L2loss = torch.tensor(0.0, device=dev)
            Linfloss = torch.tensor( 0.0, device=dev)

            # TODO (Maybe) Test Batch Gradient Descent
            self.L2optimizers[ii].zero_grad()
            for X, index in Batch:

                loss = self.target[ii](X.requires_grad_(), model, **self.var[ii])
                
                # Collect the index where the max happens
                max, jj = torch.max(loss, dim=0)
                if max > Linfloss:
                    max_index = index[jj]
                    Linfloss = max
                
                # Resample any indices that require it
                if importance_sampling:
                    previous_max = np.sqrt(squaredlosses[ii,1])
                    self.reject[ii] = find_resample(X, index, torch.sqrt(loss.data), self.reject[ii], previous_max)

                # TODO Integral Scaling
                L2loss_batch = self.integrate[ii](X, numpts, loss)
                L2loss += L2loss_batch

                # Collect gradients on L2loss of each batch
                L2loss_batch.backward()

            # Step for L2 optimization
            self.L2optimizers[ii].step()

            # Train for Linf optimization
            self.Linfoptimizers[ii].zero_grad()
            Linfloss = self.target[ii](X[max_index,:].requires_grad_(), model, **self.var[ii])
            Linfloss.backward()
            
            # Step for Linf optimization
            self.Linfoptimizers[ii].step()

            losses[ii] = [L2loss.data.cpu().numpy(), Linfloss.data.cpu().numpy()]

        return losses

    def ResamplePoints(self, domain, problem_parameters):
        input_dim = problem_parameters['input_dim']
        input_range = problem_parameters['input_range']

        for ii in range(self.numtype):
            indices = torch.arange(self.toTrain[ii].__len__())
            if any(self.reject[ii]):
                indices = indices[self.reject[ii]]
            self.toTrain[ii].location[indices,:] = self.toTrain[ii].generate_points(indices.size(0), input_dim, input_range, domain)

### Number of points along boundary
###   based on Mean Minimum Distance
def num_points_wall(N, l, V, d, d0):
    Nw = int( 4*torch.round( (l/V)**(d0/2) * N**(d0/d) ).detach().numpy() + 1 )
    return Nw

def Dirichlet_target(X, model, true, **var):
    target = SSE(model(X), true(X).detach())

    return target

def Inletoutlet_target(X, model, true):
    # (u,v,w, p)
    # Target = p - true_pressure
    UP = model(X)

    target = SSE(UP[:,-1], true(X).detach())

    return target

def find_resample(X, index, loss, resample_index, max):
    ### Rejection sampling ###
    # generate uniform( max_loss )
    # Compare, if loss(x) < uniform
    # Then resample

    check_sample = max*loss*torch.rand(X.size(0), device=X.device)

    resample_index = torch.cat( (resample_index, index[loss[:,0] < check_sample]), dim=0)

    return resample_index

# TODO
class forError():

    def __init__(self, num, domain, problem_parameters, dev):
        # Organize variables
        input_dim = problem_parameters['input_dim']
        input_range = problem_parameters['input_range']
        unsteady = bool(input_dim[1])

        self.true = problem_parameters['error']['true_fun']

         # Interior Points
        self.Points = [InteriorPoints(num, domain, input_dim, input_range, dev)]
        self.integrate = [domain.integrate]

        # domain.wall + domain.solid
        for bdry in domain.wall + domain.solid:
            nb = num_points_wall(num, bdry.measure, domain.volume, input_dim[0], bdry.dim)
            self.Points.append(BCPoints(nb, domain, bdry, input_dim, input_range, dev))
            self.integrate.append( bdry.integrate )

        # TODO domain.inletoutlet
        for inletoutlet in domain.inletoutlet:
            nb = num_points_wall(num, bdry.measure, domain.volume, input_dim[0] + input_dim[1], bdry.dim)
            self.Points.append(BCPoints(nb, inletoutlet, input_dim, input_range))
        # TODO mesh
        for mesh in domain.mesh:
            self.Points.append(MeshPoints(num_mesh, mesh))
        # TODO Unsteady problem
        if unsteady:
            self.Points.append()

        # How many points
        self.numtype = int(len(self.Points))

    def CalculateError(self, model, dev, numbatch):
        # Output: Total L2 error, Linf error

        squarederrors = np.zeros( (self.numtype,2) )

        for ii in range(self.numtype):
            # Organize into batch
            numpts = self.Points[ii].__len__()
            Batch = torch.utils.data.DataLoader(self.Points[ii], batch_size=numbatch, shuffle=True)

            L2error = torch.tensor(0.0, device=dev)
            Linferror = torch.tensor(0.0, device=dev)
            V = torch.tensor(0.0, device=dev)

            # Do in batches
            for X, index in Batch:

                Y = model(X.requires_grad_())
                Ytrue = self.true(X.requires_grad_())
                
                batch_error = SSE(Y.detach(), Ytrue.detach())
                batch_var = batch_error**2

                L2error += self.integrate[ii](X, numpts, batch_error) 
                Linferror = torch.max( Linferror, torch.max(batch_error) )

                V += self.integrate[ii](X, numpts, batch_var) 

            squarederrors[ii,:] = [L2error.data.cpu().numpy(), Linferror.data.cpu().numpy()]

        # TODO Give confidence bars using standard error
        # StandardError = torch.sqrt( (Domain.volume**2 V/numpts/(numpts-1) - Domain.volume * L2error**2/( numpts-1 ) )
        
        return squarederrors

### DataLoader classes for different types of points

class InteriorPoints(torch.utils.data.Dataset):
    ###
    ### Points that are inside the domain
    ###

    def __init__(self, num, domain, input_dim, input_range, dev):
        
        self.location = self.generate_points(num, input_dim, input_range, domain).to(dev)
        self.num_pts = num

    def generate_points(self, num, input_dim, input_range, domain):
        ### Generate points inside the domain

        X = torch.zeros( ( num, int(np.sum(input_dim)) ) )

        for ii in range(input_dim[0]):
            X[:,ii] = (input_range[ii][1] - input_range[ii][0])*torch.rand( (num) ) + input_range[ii][0]

        outside = torch.zeros( X.size(0), dtype=torch.bool)
        for wall in domain.inside:
            outside += wall.distance(X) < 0

        if any(outside):
            X[outside,:] = self.generate_points(torch.sum(outside), input_dim, input_range, domain)
            
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
    
    def __init__(self, num, domain, bdry, input_dim, input_range, dev):
        
        self.num_pts = num
        self.bdry = bdry

        self.location = self.generate_points(num, input_dim, input_range, domain).to(dev)

    def generate_points(self, nb, input_dim, input_range, domain):

        Xbdry = torch.zeros( (nb, int(sum(input_dim))))
        Xbdry[:,:input_dim[0]] = self.bdry.make_points(nb)

        # Include time
        if input_dim[1]:
            # Fill in time values
            t = input_dim[0]
            Xbdry[:,t] = (input_range[t][1] - input_range[t][0])*torch.rand( (nb) ) + input_range[t][0]
        # Include hyperparameters
        if input_dim[2]:
            # Fill in hyperparameter values
            # TODO Do exponential scaling
            hstart = input_dim[0] + input_dim[1]
            hend = sum(input_dim)
            for jj in range(hstart, hend):
                Xbdry[:,jj] = (input_range[jj][1] - input_range[jj][0])*torch.rand( (nb) ) + input_range[jj][0]

        return Xbdry

    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, index):
        return self.location[index,:], index    
    
class ICPoints(torch.utils.data.Dataset):
    ###
    ### Points that are along the initial time of the space-time domain
    ### Evaluate the initial condition at these points
    ###

    def __init__(self, num_init, domain, input_dim, input_range, dev):
        
        Xinit = self.generate_points(num_init, input_dim, input_range, domain).to(dev)
        
        self.location = Xinit
        self.num_pts = num_init

    def generate_points(num, input_dim, input_range, domain):
        ### Generate points inside the domain

        X = torch.zeros( ( num, int(np.sum(input_dim)) ) )

        for ii in range(input_dim[0]):
            X[:,ii] = (input_range[ii][1] - input_range[ii][0])*torch.rand( (num) ) + input_range[ii][0]

        outside = torch.zeros( X.size(0), dtype=torch.bool)
        for wall in domain.inside:
            outside += wall.distance(X) < 0

        if any(outside):
            X[outside,:] = generate_interior_points(torch.sum(outside), input_dim, input_range, domain)
            
        if input_dim[1]:
            # Zero out time
            t = input_dim[0]
            X[:,t] = torch.zeros((num))

        if input_dim[2]:
            # Fill in hyperparameter values
            # TODO Do exponential scaling
            hstart = input_dim[0] + input_dim[1]
            hend = sum(input_dim)
            for jj in range(hstart, hend):
                X[:,jj] = (input_range[jj][1] - input_range[jj][0])*torch.rand( (num) ) + input_range[jj][0]

        return X

    def __len__(self):
        ### How many data points are there?
        return self.num_pts
    
    def __getitem__(self, index):
        ### Gets one sample of data
        ### 
        return self.location[index,:], index

### TODO
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
    
    def generate_points(self):
        pass

    def solveU(self, b):
        U = torch.linalg.solve(self.A, b)
        return U

    def __len__(self):
        return self.num_pts
    
    def __getitem__(self, index):
        return self.location[index,:], self.value[index,:], index
