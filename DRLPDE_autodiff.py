#########   Physics-Informed Solver of Partial Differential Equations

#########   Built-in packages

import numpy as np
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

##########   Main execution function

def maintraining(param='DRLPDE_param_problem',
                 param_solver = 'DRLPDE_param_solver',
                 use_cuda='torch.cuda.is_available()'):
    
    ################# Pre-processing ##################
    
    import DRLPDE_nn
    import DRLPDE_functions.DefineDomain

    torch.set_default_dtype(torch.float64)
    
    import importlib
    
    if param=='DRLPDE_param_problem':
        DRLPDE_param = importlib.import_module("DRLPDE_param_problem")
        #print("Pre-processing: Loading parameters from default location: DRLPDE_param_problem.py")
    else:
        DRLPDE_param = importlib.import_module("." + param, package='examples')
        #print("Pre-processing: Loading parameters from " + param + '.py')

    DRLPDE_param_solver = importlib.import_module(param_solver)
        
    ### Use cuda
    dev = torch.device("cuda:0" if use_cuda else "cpu")

    ### Unpack and organize variables related to the Problem
    # TODO: Safeguard is_unsteady and pde_type
    # TODO: nn_type in DRLPDE_param_solver to choose the neural network

    boundingbox = DRLPDE_param.boundingbox
    list_of_dirichlet_boundaries = DRLPDE_param.list_of_dirichlet_boundaries
    list_of_periodic_boundaries = []
    pde_type = DRLPDE_param.pde_type
    is_unsteady = DRLPDE_param.is_unsteady
    output_dim = DRLPDE_param.output_dim

    there_are_boundaries = bool(list_of_dirichlet_boundaries)
    
    nn_param = {'depth': DRLPDE_param_solver.nn_depth,
                'width': DRLPDE_param_solver.nn_width,
                'x_dim':DRLPDE_param.x_dim,
                'is_unsteady':DRLPDE_param.is_unsteady,
                'output_dim':DRLPDE_param.output_dim
                }
    
    ### Import functions
    if is_unsteady:
        # Include time range in bounding box
        boundingbox.append(DRLPDE_param.time_range)
        init_con = DRLPDE_param.init_con
    
    ### Organize parameters related to deep learning solver
    num_walkers = DRLPDE_param_solver.num_walkers
    num_batch = DRLPDE_param_solver.num_batch
    update_walkers_every = DRLPDE_param_solver.update_walkers_every
    
    num_bdry = DRLPDE_param_solver.num_bdry
    num_batch_bdry = DRLPDE_param_solver.num_batch_bdry

    num_epoch = DRLPDE_param_solver.num_epoch
    update_print_every = DRLPDE_param_solver.update_print_every

    lambda_bell = DRLPDE_param_solver.lambda_bell
    lambda_bdry = DRLPDE_param_solver.lambda_bdry
    
    if is_unsteady:
        num_init = DRLPDE_param_solver.num_init
        num_batch_init = DRLPDE_param_solver.num_batch_init
        lambda_init = DRLPDE_param_solver.lambda_init
    

    ################ Preparing the model #################
    
    #print("Initializing the model")
    
    ### Make boundaries defining the domain
    Domain = DRLPDE_functions.DefineDomain.Domain(is_unsteady, boundingbox, 
                                                  list_of_dirichlet_boundaries,
                                                  list_of_periodic_boundaries)
    
    ### Initialize the Model
    MyNeuralNetwork = DRLPDE_nn.IncompressibleNN


    if DRLPDE_param.loadmodel:
        model = MyNeuralNetwork(**nn_param).to(dev)
        model.load_state_dict(torch.load("savedmodels/" + DRLPDE_param.loadmodel + ".pt"))
        print("Using model from savedmodels/" + DRLPDE_param.loadmodel + ".pt")
    else:
        model = MyNeuralNetwork(**nn_param).to(dev)

    mseloss = nn.MSELoss(reduction = 'mean')
    optimizer = optim.Adam(model.parameters(), 
                           lr=DRLPDE_param_solver.learning_rate, 
                           betas=DRLPDE_param_solver.adam_beta, 
                           weight_decay=DRLPDE_param_solver.weight_decay)

    ### Create Interior points: InPoints 
    ### Organize into DataLoader
    InPoints = DRLPDE_functions.DefineDomain.Walker_Data(num_walkers, boundingbox, Domain.boundaries)
    InPoints_batch = torch.utils.data.DataLoader(InPoints, batch_size=num_batch, shuffle=True)

    ResamplePoints = torch.zeros_like(InPoints.location)

    ### Create Boundary points: BPoints
    ### Organize into DataLoader
    if there_are_boundaries:
        BPoints = DRLPDE_functions.DefineDomain.Boundary_Data(num_bdry, boundingbox, Domain.boundaries, is_unsteady)
        BPoints_batch = torch.utils.data.DataLoader(BPoints, batch_size=num_batch_bdry, shuffle=True)

    if is_unsteady:
        InitPoints = DRLPDE_functions.DefineDomain.Initial_Data(num_init, boundingbox, Domain.boundaries, init_con)
        InitPoints_batch = torch.utils.data.DataLoader(InitPoints, batch_size=num_batch_init, shuffle=True)

    

    ################ Training the model #################
    
    #print("Training has begun")
    
    start_time = time.time()

    for step in range(num_epoch):

        # Interior Points
        for X, index in InPoints_batch:

            # Send to GPU and set requires grad flag
            X = X.to(dev).requires_grad_(True)
            U = model(X)

            Target = DRLPDE_nn.autodiff_vB(U, X)

            # Calculate loss at each point
            loss_everywhere = torch.norm(Target, dim=1)

            ### Rejection sampling
            # Calculate Loss at each point - not sum it up
            # Calculate the max loss
            # Sample uniformly from 0 to max loss
            # If original value is less than new value, then resample the corresponding point
            
            max_loss = torch.max(loss_everywhere).detach()
            check_sample = max_loss*torch.rand(X.size(0), device=X.device)
            resample = loss_everywhere < check_sample
            if torch.any(resample):
                Xnew = X.data.cpu()
                Xnew[resample,:] = DRLPDE_functions.DefineDomain.generate_interior_points(torch.sum(resample), boundingbox, Domain.boundaries)
                ResamplePoints[index,:] = Xnew

            ### Backwards pass
            loss = torch.mean(loss_everywhere)
            loss.backward()

        # Boundary Points
        if there_are_boundaries:
            for Xbdry, Ubtrue in BPoints_batch:
                Xbdry = Xbdry.to(dev).requires_grad_(True)
                Ubtrue = Ubtrue.to(dev).detach()
                Ubdry = model(Xbdry)
                loss = lambda_bdry*mseloss(Ubdry, Ubtrue)
                loss.backward()

        # Initial Points
        if is_unsteady:
            for Xinit, Uinit_true in InitPoints_batch:
                Xinit = Xinit.to(dev).requires_grad_(True)
                Uinit_true = Uinit_true.to(dev).detach()
                Uinit = model(Xinit)
                loss = lambda_init*mseloss(Uinit, Uinit_true)
                loss.backward()

        # Make optimization step
        optimizer.step()
        optimizer.zero_grad()

        # Update walkers
        if (step+1) % update_walkers_every == 0:
            InPoints.location = ResamplePoints
            InPoints_Batch = torch.utils.data.DataLoader(InPoints, batch_size=num_batch, shuffle=True)

        # Print statements
        if step == 0:
            print('No errors in first epoch')
            current_time = time.time()
            print('Approx time: {:.0f} minutes'.format((current_time - start_time)*num_epoch/60))
        if (step+1) % update_print_every == 0:
            current_time = time.time()
            print('step = {0}/{1}, {2:2.3f} s/step, time-to-go:{3:2.0f} min'.format(
                    step+1, num_epoch, (current_time - start_time) / (step + 1), 
                (current_time - start_time) / (step + 1) * (num_epoch - step - 1)/60))

    # Save model as pickle file
    if DRLPDE_param.savemodel:
        torch.save(model.state_dict(), "savedmodels/" + DRLPDE_param.savemodel + ".pt")
        #print("model saved as savedmodels/" + DRLPDE_param.savemodel + ".pt")

    # Return the model for plotting
    return model
   
          
if __name__ == "__main__":

    ### Main training step ###
    import argparse
    parser = argparse.ArgumentParser(description="Automatic differentiation of NNs to solve PDEs")
    parser.add_argument('-example', type=str)
    parser.add_argument('-use_cuda', type=bool)
    
    args = parser.parse_args()
    
    if args.example:
        param = args.example
    else:
        param = 'DRLPDE_param_problem'
        
    if args.use_cuda:
        use_cuda = args.use_cuda
    else:
        use_cuda = torch.cuda.is_available()
        
    #use_model = maintraining(param, use_cuda)
          
    ### Plot stuff ###
    #import DRLPDE_postprocessing
    
    #DRLPDE_postprocessing.postprocessing(param, use_model)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    