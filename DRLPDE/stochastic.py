import torch
import DRLPDE.autodiff as ad
import numpy as np
import DRLPDE.create as create

SquaredError = torch.nn.MSELoss(reduction='none')

### **var_train = {x_dim, domain, initial_con, dt, num_batch, num_ghost, tol, dev}

### TODO forcing terms
###      When doing exit condition, have to calculate exit time

### TODO higher order SDE simulation

### TODO Check if .clone().detach() is slowing down the code

### Viscous Burgers
###    Only compatible with Dirichlet BC
###    Use Incompressible NN to solve, fluid flow

def unsteadyViscousBurgers(X, model, domain, x_dim, diffusion, forcing, dt, num_ghost, tol, ic, **var_train):
    
    ### X:     (x,y,z,t,hyper)
    ### model: (u,v,w) = curl(A)
    # Xnew = X repeated
    Xnew = X.clone().detach().repeat(num_ghost,1)

    # Evaluate at X
    Uold = model(X)

    # Diffusion coefficient
    mu = diffusion(X)
    
    if mu.size():
        mu = mu.repeat(num_ghost,1)

    # Move walkers
    # TODO: Implement higher order SDE simulation
    Xnew[:,:x_dim] = Xnew[:,:x_dim] - dt*Uold.repeat(num_ghost,1) + torch.sqrt(2*dt*mu)*torch.randn((Xnew.size(0), x_dim), device=X.device, requires_grad=True)
    Xnew[:,x_dim] = Xnew[:,x_dim] - dt

    # Periodic boundaries
    if any(domain.periodic):
        Xnew[:,:x_dim] = exit_periodic(Xnew[:,:x_dim], domain.periodic)

    # Evaluate at Xnew
    Unew = model(Xnew)

    # Calculate exits and re-evaluate
    Xnew, Unew = exit_bc(X.repeat(num_ghost,1), Xnew, Unew, domain.exitflag, x_dim, tol)
    Xnew, Unew = exit_ic(X.repeat(num_ghost,1), Xnew, Unew, ic, x_dim, tol)

    # Calculate Loss = Residual Squared Error
    Loss = SquaredError( Unew.detach().reshape(num_ghost, X.size(0), x_dim).mean(0), Uold )

    return Loss

def steadyViscousBurgers(X, model, domain, x_dim, diffusion, forcing, dt, num_ghost, tol, **var_train):

    ### X: (x,y,z,hyper)
    ### model: (u,v,w) = curl(A)

    # Xnew = X repeated
    Xnew = X.clone().detach().repeat(num_ghost,1)

    # Evaluate at X
    Uold = model(X)
    
    # Diffusion coefficient
    mu = diffusion(X)

    if mu.size():
        mu = mu.repeat(num_ghost,1)

    # Move walkers
    # TODO: Implement higher order SDE simulation
    Xnew[:,:x_dim] = Xnew[:,:x_dim] - dt*Uold.detach().repeat(num_ghost,1) + torch.sqrt(2*dt*mu)*torch.randn((Xnew.size(0), x_dim), device=X.device, requires_grad=True)

    # Periodic boundaries
    if any(domain.periodic):
        Xnew[:,:x_dim] = exit_periodic(Xnew[:,:x_dim], domain.periodic)

    # Evaluate at Xnew
    Unew = model(Xnew)

    # Calculate exits and re-evaluate
    Xnew, Unew = exit_bc(X.repeat(num_ghost,1), Xnew, Unew, domain.exitflag, x_dim, tol)

    # Calculate Loss = Residual Squared Error
    Loss = SquaredError( Unew.detach().reshape(num_ghost, X.size(0), x_dim).mean(0), Uold )

    return Loss

### Full Navier-Stokes:
###    Use whenever Inlet/Outlets are used: Pressure condition at the inlet/outlet
###    Use Incompressible NN to ensure incompressibility

def unsteadyNavierStokes(X, model, domain, x_dim, diffusion, forcing, dt, num_ghost, tol, ic, **var_train):
    
    ### X: (x,y,z,t,hyper)
    ### model: (u,v,w,p) = (curl(A), p)

    # Xnew = X repeated
    Xnew = X.clones.detach().repeat(num_ghost,1)

    # Evaluate at X
    UPold = model(X)

    # Evaluate grad(p) at X
    gradPold = ad.gradient(UPold[:,-1], X[:,:x_dim])

    # Diffusion coefficient
    mu = diffusion(X)

    if mu.size():
        mu = mu.repeat(num_ghost,1)

    # Move walkers
    # TODO: Implement higher order SDE simulation
    Xnew[:,:x_dim] = Xnew[:,:x_dim] - dt*UPold.detach().repeat(num_ghost,1) + np.sqrt(2*dt*mu)*torch.randn((Xnew.size(0), x_dim), device=X.device, requires_grad=True)
    Xnew[:,x_dim] = Xnew[:,x_dim] - dt

    # Periodic boundaries
    if any(domain.periodic):
        Xnew[:,:x_dim] = exit_periodic(Xnew[:,:x_dim], domain.periodic)

    # Inlet/outlet
    if any(domain.inletoutlet):
        Xnew[:,:x_dim] = exit_inletoutlet(X.repeat(num_ghost,1), Xnew[:,:x_dim], domain.inletoutlet, x_dim, tol)

    # Evaluate at Xnew
    UPnew = model(Xnew)

    # Calculate exits and re-evaluate bc, inletoutlet, ic
    Xnew, UPnew = exit_bc(X.repeat(num_ghost,1), Xnew, UPnew, domain.exitflag, x_dim, tol)

    Xnew, UPnew = exit_ic(X.repeat(num_ghost,1), Xnew, UPnew, ic, x_dim, tol)

    # Evaluate grad(p) at Xnew
    gradPnew = ad.gradient(UPnew[:,-1], Xnew[:,:x_dim])
    
    # Calculate Loss = Residual Squared Error
    Loss = SquaredError( UPnew[:,:x_dim].detach().reshape(num_ghost, X.size(0), x_dim).mean(0) + dt/2 *( gradPnew.detach().reshape(num_ghost, X.size(0), x_dim).mean(0), UPold + gradPold) )

    return Loss

def steadyNavierStokes(X, model, domain, x_dim, diffusion, forcing, dt, num_ghost, tol, **var_train):
    
    ### X: (x,y,z,hyper)
    ### model: (u,v,w,p) = (curl(A), p)

    # Xnew = X repeated
    Xnew = X.clone.detach().repeat(num_ghost,1)

    # Evaluate at X
    UPold = model(X)

    # Evaluate grad(p) at X
    gradPold = ad.gradient(UPold[:,-1], X[:,:x_dim])

    mu = diffusion(X)

    if mu.size():
        mu = mu.repeat(num_ghost,1)

    # Move Xnew
    # TODO: Implement higher order SDE simulation
    Xnew[:,:x_dim] = Xnew[:,:x_dim] - dt*UPold.detach().repeat(num_ghost,1) + torch.sqrt(2*dt*mu)*torch.randn((Xnew.size(0), x_dim), device=X.device, requires_grad=True)

    # Periodic boundaries
    if any(domain.periodic):
        Xnew[:,:x_dim] = exit_periodic(Xnew[:,:x_dim], domain.periodic)

    # Inlet/outlet
    if any(domain.inletoutlet):
        Xnew[:,:x_dim] = exit_inletoutlet(X.repeat(num_ghost,1), Xnew[:,:x_dim], domain.inletoutlet, x_dim, tol)

    # Evaluate at Xnew
    UPnew = model(Xnew)

    # Calculate exits and re-evaluate
    Xnew, UPnew = exit_bc(X.repeat(num_ghost,1), Xnew, UPnew, domain.exitflag, x_dim, tol)

    # Evaluate grad(p) at Xnew
    gradPnew = ad.gradient(UPnew[:,-1], Xnew[:,:x_dim])

     # Calculate Loss = Residual Squared Error
    Loss = SquaredError( UPnew[:,:x_dim].detach().reshape(num_ghost, X.size(0), x_dim).mean(0) + dt/2 *( gradPnew.detach().reshape(num_ghost, X.size(0), x_dim).mean(0), UPold + gradPold) )

    return Loss

### Laplace Equation
def Laplace(X, model, domain, x_dim, diffusion, forcing, dt, num_ghost, tol, **var_train):
    ### X: (x,y,z,hyper)
    ### model: u


    # Xnew = X repeated
    Xnew = X.clone().detach().repeat(num_ghost,1)

    # Evaluate at X
    Uold = model(X)
    
    # Diffusion coefficient
    mu = diffusion(X)

    if mu.size():
        mu = mu.repeat(num_ghost,1)

    # Move walkers
    # TODO: Implement higher order SDE simulation
    Xnew[:,:x_dim] = Xnew[:,:x_dim] + torch.sqrt(2*dt*mu)*torch.randn((Xnew.size(0), x_dim), device=X.device, requires_grad=True)

    # Periodic boundaries
    if any(domain.periodic):
        Xnew[:,:x_dim] = exit_periodic(Xnew[:,:x_dim], domain.periodic)

    # Evaluate at Xnew
    Unew = model(Xnew)

    # Calculate exits and re-evaluate
    Xnew, Unew = exit_bc(X.repeat(num_ghost,1), Xnew, Unew, domain.exitflag, x_dim, tol)

    # Make target
    Loss = SquaredError( Unew.detach().reshape(num_ghost, X.size(0), Uold.size(1)).mean(0), Uold)

    return Loss

### Heat Equation
def Heat(X, model, domain, x_dim, diffusion, forcing, dt, num_ghost, tol, ic, **var_train):
    ### X: (x,y,z,hyper)
    ### model: u

    # Xnew = X repeated
    Xnew = X.clone().detach().repeat(num_ghost,1)

    # Evaluate at X
    Uold = model(X)
    
    # Diffusion coefficient
    mu = diffusion(X)

    if mu.size():
        mu = mu.repeat(num_ghost,1)

    # Move walkers
    # TODO: Implement higher order SDE simulation
    Xnew[:,:x_dim] = Xnew[:,:x_dim] + torch.sqrt(2*dt*mu)*torch.randn((Xnew.size(0), x_dim), device=X.device, requires_grad=True)
    Xnew[:,x_dim] = Xnew[:,x_dim] - dt

    # Periodic boundaries
    if any(domain.periodic):
        Xnew[:,:x_dim] = exit_periodic(Xnew[:,:x_dim], domain.periodic)

    # Evaluate at Xnew
    Unew = model(Xnew)

    # Calculate exits and re-evaluate
    Xnew, Unew = exit_bc(X.repeat(num_ghost,1), Xnew, Unew, domain.exitflag, x_dim, tol)

    Xnew, Unew = exit_ic(X.repeat(num_ghost,1), Xnew, Unew, ic, x_dim, tol)

    # Make target
    Loss = SquaredError( Unew.detach().reshape(num_ghost, X.size(0), Uold.size(1)).mean(0), Uold)

    return Loss

### TODO: Elliptic Equation
def steadyElliptic(X, model, mu, x_dim, domain, drift, dt, num_batch, num_ghost, tol, **var_train):
    
    # Xnew = X repeated
    Xnew = X.repeat(num_ghost,1)

    # Evaluate at X
    Uold = model(X)

    # Move Xnew
    Xnew = X.repeat(num_ghost,1) - dt*drift(X).detach().repeat(num_ghost,1) + torch.sqrt(2*mu*dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)
    
    # Calculate periodic boundaries
    Xnew = exit_periodic(Xnew, domain.periodic)

    # Evaluate at Xnew
    Unew = model(Xnew)

    # Calculate exits
    Xnew, Unew = exit_bc(X.repeat(num_ghost,1), Xnew, Unew, domain.exitflag, tol)

    # Make target
    Target = Unew.detach().reshape(num_ghost, num_batch, x_dim).mean(0) - Uold
    
    return Target

### TODO: Parabolic Equation
def unsteadyParabolic(X, model, domain, mu, x_dim, dt, num_batch, num_ghost, tol, drift, **var_train):
    
    ### Evaluate model
    Uold = model(X)

    Xnew = X.clone().detach().repeat(num_ghost,1)

    ### Move walkers
    Zt = np.sqrt(dt)*torch.randn((num_batch*num_ghost, x_dim), device=X.device, requires_grad=True)

    Xnew[:,:x_dim] = Xnew[:,:x_dim] - dt*drift(X).detach().repeat(num_ghost,1) + np.sqrt(2*mu)*Zt                                         
    Xnew[:,x_dim] = Xnew[:,x_dim] - -dt*torch.ones((num_batch*num_ghost,1), device=X.device)

    ### Calculate exits
    Xnew, outside = exit_bc(X.repeat(num_ghost,1), Xnew, domain.exitflag, tol)

    ### Calculate periodic boundaries
    if any(domain.periodic_boundaries):
        Xnew = exit_periodic(Xnew, domain.periodic_boundaries)
    
    #Loss = SquaredError

    return Xnew, Uold, outside[:num_batch]


### Simulate Stochastic Process

def walk(X, num, model, input_dim, input_range, diffusion, x_dim, domain, dt, **var_train):
    
    # Evaluate at X
    Uold = model(X)

    # Diffusion coefficient
    mu = diffusion(X).detach()
    
    # Move walkers
    # TODO: Implement higher order SDE simulation
    Xnew = X[:,:x_dim] - dt*Uold + torch.sqrt(2*dt*mu)*torch.randn((num, x_dim), device=X.device, requires_grad=False)

    # Periodic boundaries
    if any(domain.periodic):
        Xnew = exit_periodic(Xnew[:,:x_dim], domain.periodic)

    for bdry in domain.inside:
        outside_bdry = bdry.distance(Xnew[:,:x_dim]) < 0
        if any(outside_bdry):
            Xnew[outside_bdry,:x_dim] = create.generate_interior_points(torch.sum(outside_bdry), input_dim, input_range, domain, domain.inside)

    return Xnew.detach()

def move_steadySDE():

    pass

def move_unsteadySDE():

    pass

### Exit Calculations

def exit_bc(Xold, Xnew, Unew, boundary, x_dim, tol):
    ### Calculate boundary exits and enforce boundary condition
    
    # If you want to record which walkers exited
    #outside = torch.zeros( Xnew.size(0), dtype=torch.bool, device=Xnew.device)
    
    for bdry in boundary:
        
        outside_bdry = bdry.distance(Xnew[:,:x_dim]) < 0
        if torch.sum(outside_bdry) > 0:
            ### Recursive bisection to get close to exit location up to tolerance tol

            Xnew[outside_bdry,:x_dim] = find_bdry_exit(Xold[outside_bdry,:x_dim], Xnew[outside_bdry,:x_dim], bdry, tol)
            Unew[outside_bdry,:] = bdry.bc(Xnew[outside_bdry,:])
            
        #outside += outside_bdry

    return Xnew, Unew

def exit_ic(Xold, Xnew, Unew, initial_con, x_dim, tol):
    ### Check for time = 0
    ###     Should we take a point at the initial time (by projecting or something)
    ###     or is within tol good enough?

    # If you want to record which walkers exited
    #outside = torch.zeros( Xnew.size(0), dtype=torch.bool, device=Xnew.device)
    
    hit_initial = Xnew[:,x_dim] < 0
    Xnew[hit_initial,:] = find_time_exit(Xold[hit_initial,:], Xnew[hit_initial,:], tol)
    Unew[hit_initial,:] = initial_con(Xnew[hit_initial,:])

    #outside += hit_initial

    return Xnew, Unew 

def exit_inletoutlet(Xold, Xnew, inletoutlet, tol):
    ### Calculate inlet/outlet exits
    
    # If you want to record which walkers exited
    #outside = torch.zeros( Xnew.size(0), dtype=torch.bool, device=Xnew.device)


    for bdry in inletoutlet:
        outside_bdry = bdry.distance(Xnew) < 0
        if torch.sum(outside_bdry) > 0:
            ### Bisection to get close to exit location up to tolerance tol
            Xnew[outside_bdry,:] = find_bdry_exit(Xold[outside_bdry,:], Xnew[outside_bdry,:], bdry, tol)
        
        #outside += outside_bdry
    return Xnew

def find_bdry_exit(Xold, Xnew, bdry, tol):
    ### Bisection algorithm to find the exit between Xnew and Xold up to a tolerance 
    
    Xmid = (Xnew + Xold)/2
    
    dist = bdry.distance(Xmid)
    
    # above tolerance = inside
    # below -tolerance = outside
    inside = dist > tol
    outside = dist < -tol
    
    if torch.sum(inside + outside) > 0:
        Xnew[outside,:] = Xmid[outside,:]
        Xold[inside,:] = Xmid[inside,:]
        Xmid[inside + outside,:] = find_bdry_exit(Xold[inside + outside,:], Xnew[inside + outside,:], bdry, tol)

    return Xmid

def find_time_exit(Xold, Xnew, tol):
    ### Bisection algorithm to find the time exit up to a tolerance
    
    Xmid = (Xnew + Xold)/2

    # above tolerance = inside
    # below tolerance = outside
    above_tol = Xmid[:,2] > tol
    below_tol = Xmid[:,2] < -tol

    if torch.sum(above_tol + below_tol) > 0:
        Xnew[below_tol,:] = Xmid[below_tol,:]
        Xold[above_tol,:] = Xmid[above_tol,:]
        
        Xmid[above_tol + below_tol,:] = find_time_exit(Xold[above_tol + below_tol,:], Xnew[above_tol + below_tol,:], tol)

    return Xmid

def exit_periodic(Xnew, periodic_boundaries):
    
    for bdry in periodic_boundaries:
        below_bot = Xnew[:,bdry.index] < bdry.bot
        above_top = Xnew[:,bdry.index] > bdry.top

        if torch.sum(below_bot) > 0:
            Xnew[below_bot, bdry.index] = Xnew[below_bot, bdry.index] + (bdry.top - bdry.bot)
        if torch.sum(above_top) > 0:
            Xnew[above_top,bdry.index] = Xnew[above_top, bdry.index] - (bdry.top - bdry.bot)
    return Xnew
