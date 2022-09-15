###
### This module contains the functions moving and evaluating the walkers
###

### TODO: Add functions that perform finite difference operations


import torch
import math
import numpy as np

### Move Walkers
def move_Walkers_NS_steady(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    
    ### Evaluate model
    Uold = model(X)

    ### Move walkers
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1) - dt*Uold.detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt

    ### Calculate exits
    Xnew, outside = exit_condition_steady(X.repeat(num_ghost,1), Xnew, Domain.boundaries, tol)

    ### Calculate periodic boundaries
    Xnew = periodic_condition(Xnew, Domain.periodic_boundaries)

    return Xnew, Uold, outside[:num_batch]

def move_Walkers_NS_unsteady(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    
    ### Evaluate model
    Uold = model(X)

    ### Move walkers
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1)  + torch.cat( (-dt*Uold.detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt, 
                                                -dt*torch.ones((num_batch*num_ghost,1), device=X.device, requires_grad=True)), dim=1)
    
    ### Calculate exits
    Xnew, outside = exit_condition_unsteady(X.repeat(num_ghost,1), Xnew, Domain.boundaries, tol)
    
    ### Calculate periodic boundaries
    Xnew = periodic_condition(Xnew, Domain.periodic_boundaries)

    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Stokes_steady(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    
    ### Evaluate model
    Uold = model(X)

    ### Move walkers
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1) + np.sqrt(2*mu)*Zt
    
    ### Calculate exits
    Xnew, outside = exit_condition_steady(X.repeat(num_ghost,1), Xnew, Domain.boundaries, tol)
    
    ### Calculate periodic boundaries
    Xnew = periodic_condition(Xnew, Domain.periodic_boundaries)

    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Stokes_unsteady(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, **move_walkers_param):
    
    ### Evaluate model
    Uold = model(X)

    ### Move walkers
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1)  + torch.cat( ( np.sqrt(2*mu)*Zt, 
                                                -dt*torch.ones((num_batch*num_ghost,1), device=X.device, requires_grad=True)), dim=1)
    
    ### Calculate exits
    Xnew, outside = exit_condition_unsteady(X.repeat(num_ghost,1), Xnew, Domain.boundaries, tol)

    ### Calculate periodic boundaries
    Xnew = periodic_condition(Xnew, Domain.periodic_boundaries)

    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Elliptic(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, drift, **move_walkers_param):
    
    ### Evaluate model
    Uold = model(X)

    ### Move walkers
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    Xnew = X.repeat(num_ghost,1) - dt*drift(X).detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt
    
    ### Calculate exits
    Xnew, outside = exit_condition_steady(X.repeat(num_ghost,1), Xnew, Domain.boundaries, tol)

    ### Calculate periodic boundaries
    Xnew = periodic_condition(Xnew, Domain.periodic_boundaries)
    
    return Xnew, Uold, outside[:num_batch]

def move_Walkers_Parabolic(X, model, Domain, x_dim, mu, dt, num_batch, num_ghost, tol, drift, **move_walkers_param):
    
    ### Evaluate model
    Uold = model(X)

    ### Move walkers
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)

    Xnew = X.repeat(num_ghost,1)  + torch.cat( (-dt*drift(X).detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt, 
                                                -dt*torch.ones((num_batch*num_ghost,1), device=X.device, requires_grad=True)), dim=1)
    
    ### Calculate exits
    Xnew, outside = exit_condition_unsteady(X.repeat(num_ghost,1), Xnew, Domain.boundaries, tol)

    ### Calculate periodic boundaries
    Xnew = periodic_condition(Xnew, Domain.periodic_boundaries)
    
    return Xnew, Uold, outside[:num_batch]

### Find exit location
def exit_condition_steady(Xold, Xnew, boundaries, tol):
    ### Calculate exit conditions
    outside = torch.zeros( Xnew.size(0), dtype=torch.bool, device=Xnew.device)
    
    for bdry in boundaries:
        outside_bdry = bdry.dist_to_bdry(Xnew) < 0
        if torch.sum(outside_bdry) > 0:
            ### Bisection to get close to exit location
            ### TODO: should we take a point on the boundary (by projecting or something)
            
            Xnew[outside_bdry,:] = find_bdry_exit(Xold[outside_bdry,:], Xnew[outside_bdry,:], bdry, tol)

        outside += outside_bdry

    return Xnew, outside
    
def exit_condition_unsteady(Xold, Xnew, boundaries, tol):
    ### Calculate exit conditions
    outside = torch.zeros( Xnew.size(0), dtype=torch.bool, device=Xnew.device)
    
    for bdry in boundaries:
        outside_bdry = bdry.dist_to_bdry(Xnew) < 0
        if torch.sum(outside_bdry) > 0:
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
    
    # above tolerance = inside
    # below tolerance = outside
    above_tol = dist > tol
    below_tol = dist < -tol
    
    if torch.sum(above_tol + below_tol) > 0:
        Xnew[below_tol,:] = Xmid[below_tol,:]
        Xold[above_tol,:] = Xmid[above_tol,:]
        
        Xmid[above_tol + below_tol,:] = find_bdry_exit(Xold[above_tol + below_tol,:], Xnew[above_tol + below_tol,:], bdry, tol)

    return Xmid
            
def find_time_exit(Xold, Xnew, tol):
    ### Bisection algorithm to find the time exit up to a tolerance
    
    Xmid = (Xnew + Xold)/2

    # above tolerance = inside
    # below tolerance = outside
    above_tol = Xmid[:,-1] > tol
    below_tol = Xmid[:,-1] < -tol

    if torch.sum(above_tol + below_tol) > 0:
        Xnew[below_tol,:] = Xmid[below_tol,:]
        Xold[above_tol,:] = Xmid[above_tol,:]
        
        Xmid[above_tol + below_tol,:] = find_time_exit(Xold[above_tol + below_tol,:], Xnew[above_tol + below_tol,:], tol)

    return Xmid

### Update at periodic boundaries
def periodic_condition(Xnew, periodic_boundaries):
    for bdry in periodic_boundaries:
        below_base = Xnew[:,bdry.index] < bdry.base
        above_top = Xnew[:,bdry.index] > bdry.top

        if torch.sum(below_base) > 0:
            Xnew[below_base, bdry.index] = Xnew[below_base, bdry.index] + (bdry.top - bdry.base)
        if torch.sum(above_top) > 0:
            Xnew[above_top,bdry.index] = Xnew[above_top, bdry.index] - (bdry.top - bdry.base)
    return Xnew

### Evaluate Model at new location
def evaluate_model_NS(Xold, Xnew, model, dt, forcing, **eval_model_param):
    Target = model(Xnew) + (forcing(Xold) + forcing(Xnew))*dt/2

    return Target
    
def evaluate_model_PDE(Xold, Xnew, model, dt, forcing, reaction, **eval_model_param):
    Target = model(Xnew)*torch.exp( reaction(Xnew)*dt) + (forcing(Xold) + forcing(Xnew))*dt/2
    
    return Target