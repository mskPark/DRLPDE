import torch

CalculateSquaredError = torch.nn.MSELoss(reduction='none')

def CalculateError(Batch, numpts, volume, model, true_fun, dev):
    # Batch: X, index

    # Output: Total L2 loss, Linfloss

    # Indices to resample
    SE = torch.tensor(0.0, device=dev)
    Linferror = torch.tensor(0.0, device=dev)
    SVar = torch.tensor(0.0, device=dev)

    # Do in batches
    for X, index in Batch:
        #X.to(dev).requires_grad_(True)

        Y = model(X.to(dev).requires_grad_(True))
        Ytrue = true_fun(X.to(dev).requires_grad_(True))
        
        batch_error = CalculateSquaredError(Y.detach(), Ytrue.detach())
        batch_var = batch_error**2

        SE += torch.sum(batch_error)
        Linferror = torch.max( Linferror, torch.sqrt( torch.max(batch_error) ))

        SVar += torch.sum(batch_var)

    L2error = volume*SE.detach()/numpts

    # Standard error TODO
    # StandardError = torch.sqrt( (Domain.volume**2 SVar/numpts/(numpts-1) - Domain.volume * SE**2/( numpts-1 ) )
    
    return L2error, Linferror