import torch

SquaredError = torch.nn.MSELoss(reduction='none')

### Derivative Operators through Automatic Differentiation 
### 
### dim_in = space_dim + time_dim
###    When functions called, should pass in x[:,:dim_in]
###    This is to accomodate neural networks having hyperparameters

def gradient(y,x):
    ### Calculates the gradient of (y) wrt (x)
    ### x - torch vector (Npoints x dim_in)
    ### y - torch scalar (Npoints)
    ### gradient - torch vector (Npoints x dim_in)
    ###
    ### CARE: If (y) vector, then gradient will be sum of gradients of y_i

    gradient = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), 
                                   create_graph=True, retain_graph=True)[0]

    return gradient
    
def jacobian(y,x):
    ### Calculates the Jacobian dy/dx of y = f(x)
    ### x - torch vector (Npoints x dim_in)
    ### y - torch vector (Npoints x dim_out)
    ### J - torch tensor (Npoints x dim_out x dim_in)

    J = torch.empty( (x.size(0), y.size(1), x.size(1) ), device=x.device)

    for ii in range(y.size(1)):
        J[:,ii,:] = gradient(y[:,ii],x)

    return J

def divergence(y,x):
    ### Calculates the Divergence
    ### Need dim_in = dim_out
    ### x - torch vector (Npoints x dim_in)
    ### y - torch vector (Npoints x dim_out)
    ### div - torch tensor (Npoints x dim_out)

    div = torch.empty(x.size(0), device=x.device)

    for ii in range(y.size(1)):
        div += gradient(y[:,ii],x)[:,ii]

    return div

def laplacian(y,x,J):
    ### Calculates the Laplacian for each component
    ### x - torch vector (Npoints x dim_in)
    ### y - torch vector (Npoints x dim_out)
    ### lap - torch tensor (Npoints x dim_out)

    lap = torch.empty(x.size(0), y.size(1), device=x.device)

    for ii in range(y.size(1)):
        grad = J[:,ii,:y.size(1)]
        lap[:,ii] = divergence(grad,x)
    return lap

def curl2D(y,x):
    ### Calculates the 2D curl operation ( f(x,y),g(x,y) ) -> dgdy - dfdx
    ### x - torch vector (Npoints x dim_in )
    ### y - torch vector (Npoints x 1 )
    ### curly - torch vector (Npoints x 2)
    ### Derivatives taken wrt first 2 inputs

    dydx = torch.autograd.grad(y, x, grad_outputs = torch.ones_like(y), 
                                        create_graph = True, retain_graph = True)[0]
    curly = torch.stack([dydx[:,1], -dydx[:,0]] , dim=1)

    return curly

def curl3D(y,x):
    ### Calculates 3D curl operation
    ### x - torch vector (Npoints x dim_in)
    ### y - torch vector (Npoints x 3)
    ### curly - torch vector (Npoints x 3)
    ### Derivatives taken wrt first 3 inputs

    e = torch.eye(3, device=x.device)

    dy0dx = torch.autograd.grad(y, x, grad_outputs=e[0,:].repeat(x.size(0), 1), 
                                create_graph=True, retain_graph = True)[0]
    dy1dx = torch.autograd.grad(y, x, grad_outputs=e[1,:].repeat(x.size(0), 1),
                                create_graph=True, retain_graph = True)[0]
    dy2dx = torch.autograd.grad(y, x, grad_outputs=e[2,:].repeat(x.size(0), 1),
                                create_graph=True, retain_graph = True)[0]

    curly = torch.stack([dy2dx[:,1] - dy1dx[:,2], dy0dx[:,2] - dy2dx[:,0], dy1dx[:,0] - dy0dx[:,1] ], dim=1)    

    return curly

### Velocity Vector Potential + Pressure Gradient Formulation
### (A,p) -> ( curl A, grad p) = (u, grad p)

def curlgrad2D(Ap,x):
    ### Calculates 2D curl and 2D grad
    ### x - torch vector (Npoints x dim_in)
    ### Ap - torch vector (Npoints x 2)
    ### curlAgradp - torch vector (Npoints x 3)

    e = torch.eye(2, device=x.device)

    dAdx = torch.autograd.grad(Ap, x, grad_outputs=e[0,:].repeat(x.size(0), 1), 
                                create_graph=True, retain_graph = True)[0]

    dpdx = torch.autograd.grad(Ap, x, grad_outputs=e[1,:].repeat(x.size(0), 1),
                                create_graph=True, retain_graph = True)[0]

    curlAgradp = torch.stack([dAdx[:,1], -dAdx[:,0], dpdx ], dim=1)

    return curlAgradp

def curlgrad3D(Ap,x):
    ### Calculates 3D curl and 3D grad
    ### x - torch vector (Npoints x dim_in)
    ### Ap - torch vector (Npoints x 4)
    ### curlAgradp - torch vector (Npoints x 6)

    e = torch.eye(4, device=x.device)

    dA0dx = torch.autograd.grad(Ap, x, grad_outputs=e[0,:].repeat(x.size(0), 1), 
                                create_graph=True, retain_graph = True)[0]
    dA1dx = torch.autograd.grad(Ap, x, grad_outputs=e[1,:].repeat(x.size(0), 1),
                                create_graph=True, retain_graph = True)[0]
    dA2dx = torch.autograd.grad(Ap, x, grad_outputs=e[2,:].repeat(x.size(0), 1),
                                create_graph=True, retain_graph = True)[0]
    dpdx = torch.autograd.grad(Ap, x, grad_outputs=e[3,:].repeat(x.size(0), 1),
                                create_graph=True, retain_graph = True)[0]                     

    curlAgradp = torch.stack([dA2dx[:,1] - dA1dx[:,2], dA0dx[:,2] - dA2dx[:,0], dA1dx[:,0] - dA0dx[:,1], dpdx ], dim=1)   
     
    return curlAgradp

def advection(y,x, J):
    ### Calculates advection in NS
    ### x - torch vector (Npoints x dim_in)
    ### y - torch vector (Npoints x dim_out)
    ### advec - torch tensor (Npoints x dim_out)
    ### Care: y is strictly velocity, dim_out = 2 or 3

    advec = torch.empty(x.size(0), y.size(1), device=x.device)
    for ii in range(y.size(1)):
        advec[:,ii] = torch.sum( J[:,ii,:y.size(1)] * y, dim=1)
    return advec

### PDEs - Return Loss = Residual Squared Error

def unsteadyViscousBurgers(x, model, diffusion, forcing, x_dim, **var_train):
    ### Calculates unsteady viscous Burgers equation for y
    ### x - torch vector (Npoints x input_dim)
    ### y - torch vector (Npoints x space_dim)

    xt_dim = x_dim +1
    y = model(x)
    mu = diffusion(x)
    f = forcing(x)

    J = jacobian(y,x)
    L = laplacian(y,x, J[:,:x_dim,:])
    A = advection(y,x[:,:x_dim], J[:,:x_dim,:])
    T = J[:,:,x_dim]

    vB = T + A - mu*L - f

    Loss = SquaredError(vB, torch.zeros_like(vB))

    return Loss

def steadyViscousBurgers(x, model, diffusion, forcing, x_dim, **var_train):
    ### Calculates steady viscous Burgers equation for y
    ### x - torch vector (Npoints x space_dim)
    ### y - torch vector (Npoints x space_dim)
    y = model(x)
    mu = diffusion(x)
    f = forcing(x)

    J = jacobian(y,x[:,:x_dim])
    L = laplacian(y,x[:,:x_dim],J)
    A = advection(y,x[:,:x_dim],J)

    vB = A - mu*L - f

    Loss = SquaredError(vB, torch.zeros_like(vB))

    return Loss

def unsteadyNavierStokes(x, model, diffusion, forcing, x_dim, **var_train):
    ### Calculates Navier-Stokes equation for y = ( curl(A), p )
    ### x - torch vector (Npoints x space_dim + 1 )
    ### y - torch vector (Npoints x space_dim + 1)
    
    xt_dim = x_dim +1
    y = model(x)
    mu = diffusion(x)
    f = forcing(x)

    J = jacobian(y, x[:,:xt_dim])
    L = laplacian(  y[:,:x_dim], x[:,:x_dim], J[:,:x_dim,:])
    A = advection(y[:,:x_dim], x[:,:x_dim],J[:,:x_dim,:])
    T = J[:,:x_dim,-1]

    NS = T + A + J[:,-1,:x_dim] - mu*L - f

    Loss = SquaredError(NS, torch.zeros_like(NS))

    return Loss

def steadyNavierStokes(x, model, diffusion, forcing, x_dim, **var_train):
    ### Calculates Navier-Stokes equation for y = ( curl(A), p )
    ### x - torch vector (Npoints x space_dim )
    ### y - torch vector (Npoints x space_dim + 1)
    
    y = model(x)
    mu = diffusion(x)
    f = forcing(x)

    J = jacobian(y, x)
    L = laplacian(  y[:,:x_dim], x, J[:,:x_dim,:])
    A = advection(y[:,:x_dim], x, J[:,:x_dim,:])

    NS = A + J[:,-1,:] - mu*L - f

    Loss = SquaredError(NS, torch.zeros_like(NS))

    return Loss

def Laplace(x, model, diffusion, forcing, x_dim, **var_train):

    y = model(x)
    mu = diffusion(x)
    f = forcing(x)
    J = jacobian(y,x)
    L = laplacian(y,x, J[:,:x_dim,:])

    Lap = mu*L - f

    Loss = SquaredError(Lap, torch.zeros_like(Lap))

    return Loss
