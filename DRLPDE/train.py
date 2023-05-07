# Module containing functions related to training the neural network

import torch

SquaredError = torch.nn.MSELoss(reduction='none')

#Closure: We don't require gradient everytime
    #            Wrap optimization routine into 
    #            def closure():

def closure():
    if torch.is_grad_enabled():
        optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    if loss.requires_grad:
        loss.backward()
    return loss

# And then perform the optimizer step by
# optimizer.step(closure)

# TODO datatype int64 may be overkill for resample index

def interior(Batch, numpts, model, make_target, var_train, dev, domainvolume, weight, max_loss, importance_sampling):
    # For interior points
    # Batch: X, index

    # Output: Total L2 loss, Linf loss

    # Indices to resample
    resample_index = torch.tensor([], dtype=torch.int64)
    L2loss = torch.tensor(0.0, device=dev)
    Linfloss = torch.tensor(0.0, device=dev)

    # Do in batches
    for X, index in Batch:
        #X.to(dev).requires_grad_(True)
        loss = make_target(X.to(dev).requires_grad_(True), model, **var_train)
        if importance_sampling:
            resample_index = find_resample(X, index, loss, resample_index, max_loss)

        L2loss += torch.sum(loss)
        Linfloss = torch.max( Linfloss, torch.sqrt( torch.max(loss).data ))

        # Multiply by volume
        loss = domainvolume*weight*torch.mean(loss)
        loss.backward()

    L2loss = domainvolume*L2loss.detach()/numpts

    return L2loss, Linfloss, resample_index

def boundary(Batch, numpts, model, make_target, dev, weight, max_loss, do_resample):
    # For boundary/initial value points
    # Batch: X, Utrue, index

    # Output: Total L2 loss, Linf loss
    #    loss = sample mean of expected value (integral over the region)

    # Initialize indices for resampling
    resample_index = torch.tensor([], dtype=torch.int64)
    L2loss = torch.tensor(0.0, device=dev)
    Linfloss = torch.tensor(0.0, device=dev)

    for Xb, Ubtrue, index in Batch:
        #Xb.to(dev).requires_grad_(True)

        loss = make_target(Xb.to(dev).requires_grad_(True), model, Ubtrue.to(dev).requires_grad_(True))
        if do_resample:
            resample_index = find_resample(Xb, index, loss, resample_index, max_loss)

        L2loss += torch.sum(loss)
        Linfloss = torch.max( Linfloss, torch.sqrt( torch.max(loss).data ))

        loss = weight*torch.mean( loss )
        loss.backward()

    L2loss = L2loss.detach()/numpts

    return L2loss, Linfloss, resample_index

def find_resample(X, index, loss, resample_index, max_loss):
    ### Rejection sampling ###
    # generate uniform( max_loss )
    # Compare sample < uniform
    # Then resample
    check_sample = max_loss*torch.rand(X.size(0), device=X.device)

    resample_index = torch.cat( (resample_index, index[loss[:,0] < check_sample]))

    return resample_index

def reweight(loss_max, stepsize):
    ### Gives a new weight to the boundary or initial condition losses
    ### Ensures that all losses are within the same order of magnitude
    ###    TODO: Maybe fine tuning, loss of main is 10**2 * loss of others
    weight =  stepsize*loss_max.detach()

    return weight

def Dirichlet_target(X, model, true):
    target = SquaredError(model(X), true.detach())

    return target

def Direct_target(X, model, true_fun):
    target = SquaredError(model(X), true_fun(X).detach())

    return target

def Inletoutlet_target(X, model, true):
    # (u,v,w, p)
    # Target = p - true_pressure
    UP = model(X)

    target = SquaredError(UP[:,-1], true.detach())

    return target