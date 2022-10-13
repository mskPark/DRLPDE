###
### This module contains the training step
###

import torch
import numpy as np
import math


for step in numsteps
    train_interior(batch)

    train_bdry(dataloader)

### Calculate Target
# Should be different for each type of method
# Should be imported from here 

### Calculate loss
# Input: Batch X
# Output: averaged loss, max of loss
# Do loss.backward() outside?

def train_interior(Batch, model, max_loss, resample_index, do_resample=False):

    ### Indices to resample
    ### dtype might be unnecessarily big
    resample_index = torch.tensor([], dtype=torch.int64)

    for X, index in Batch:
        X.to(dev).requires_grad_(True)
        U = model(X)

        Target = DRLPDE_nn.autodiff_vB(U, X)

        loss = torch.norm(Target, dim=1)

        if do_resample:
            ### Rejection sampling ###
            # generate uniform( max_loss )
            # Compare sample < uniform
            # Then resample
            check_sample = max_loss*torch.rand(X.size(0), device=X.device)
            resample_index = torch.cat( (resample_index, index[loss < check_sample]))

            ### Recalculate max loss
            max_loss = torch.max( max_loss, torch.sqrt( torch.max(loss).data ))

        loss = torch.mean( loss )
    
    return loss, max_loss, resample_index

    loss.backward()




# Send to GPU and set requires grad flag
Xold = Xold.to(dev).requires_grad_(True)

# Evaluate at old location and Move walkers
Xnew, Uold, outside = move_Walkers(Xold, model, Domain, **move_walkers_param)

# Evaluate at new location and average
Target = evaluate_model(Xold.repeat(num_ghost,1), Xnew, model, **eval_model_param).reshape(num_ghost, 
                                                                    num_batch,
                                                                    output_dim).mean(0)

# Calculate loss
loss = lambda_bell*mseloss(Uold, Target.detach())
loss.backward()

### Finite Difference
Xstencil = make_stencil()
# Input: spacing h, time discretization t, boundaries

Target = eval_on_stencil()
# Input: Type of discretization




### Boundary training
# Input: batched Xbdry, batched Ubdry
# Can include initial value training