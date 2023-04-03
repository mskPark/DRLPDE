# Module containing functions related to training the neural network

import torch

SquaredError = torch.nn.MSELoss(reduction='none')

# TODO datatype int64 may be overkill for resample index

def interior(Batch, numpts, model, make_target, var_train, dev, weight, max_loss, do_resample=False):
    # For interior points
    # Batch: X, index

    # Output: Total L2 loss, Linf loss

    # Indices to resample
    resample_index = torch.tensor([], dtype=torch.int64)
    L2loss = torch.tensor(0.0)
    Linfloss = torch.tensor(0.0)

    # Do in batches
    for X, index in Batch:
        X.to(dev).requires_grad_(True)

        loss = make_target(X, model, **var_train)
        
        if do_resample:
            resample_index = find_resample(X, index, loss, resample_index, max_loss)

        L2loss += torch.sum(loss)
        Linfloss = torch.max( Linfloss, torch.sqrt( torch.max(loss).data ))

        loss = weight*torch.mean(loss)
        loss.backward()

    L2loss = L2loss.detach()/numpts

    return L2loss, Linfloss, resample_index

def boundary(Batch, numpts, model, make_target, dev, weight, max_loss, do_resample):
    # For boundary/initial value points
    # Batch: X, Utrue, index

    # Output: Total L2 loss, Linf loss
    #    loss = sample mean of expected value (integral over the region)

    # Initialize indices for resampling
    resample_index = torch.tensor([], dtype=torch.int64)
    L2loss = torch.tensor(0.0)
    Linfloss = torch.tensor(0.0)

    for Xb, Ubtrue, index in Batch:
        Xb.to(dev).requires_grad_(True)

        loss = make_target(Xb, model, Ubtrue)

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
    resample_index = torch.cat( (resample_index, index[loss < check_sample]))

    return resample_index

def reweight(loss_main, loss_aux):
    ### Gives a new weight to the boundary or initial condition losses
    ### Ensures that all losses are within the same order of magnitude

    weight = 10**(torch.round( torch.log10(loss_main/loss_aux) ))

    return weight

def Dirichlet_target(X, model, true):
    target = SquaredError(model(X), true.detach())

    return target

def Inletoutlet_target(X, model, true):
    UP = model(X)

    target = SquaredError(UP[:,-1], true.detach())

    return target

def L2error(Batch, numpts, model, true_fun):
    # Batch: X, index

    # Output: Total L2 loss, Linfloss

    # Indices to resample
    L2error = torch.tensor(0.0)
    Linferror = torch.tensor(0.0)

    # Do in batches
    for X, index in Batch:
        X.requires_grad_(True)

        Y = model(X)
        Ytrue = true_fun(X)
        
        error = SquaredError(Y, Ytrue)
        
        L2error += torch.sum(error)
        Linferror = torch.max( Linferror, torch.sqrt( torch.max(error).data ))

    L2error = L2error.detach()/numpts

    return L2error, Linferror