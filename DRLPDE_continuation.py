#########   Continuation Method for solving non-linear Partial Differential Equations

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

    move_walkers_param={'x_dim': DRLPDE_param.x_dim,
                        'mu': DRLPDE_param.mu,
                        'dt': DRLPDE_param_solver.dt,
                        'num_batch': DRLPDE_param_solver.num_batch,
                        'num_ghost': DRLPDE_param_solver.num_ghost,
                        'tol': DRLPDE_param_solver.tol
                       }
    
    eval_model_param={'dt': DRLPDE_param_solver.dt,
                      'forcing': DRLPDE_param.forcing}
    
    ### Import functions
    if is_unsteady:
        # Include time range in bounding box
        boundingbox.append(DRLPDE_param.time_range)
        init_con = DRLPDE_param.init_con
        
        if pde_type == 'NavierStokes':
            move_Walkers = DRLPDE_functions.EvaluateWalkers.move_Walkers_NS_unsteady
        elif pde_type == 'Parabolic':
            move_Walkers = DRLPDE_functions.EvaluateWalkers.move_Walkers_Parabolic
        elif pde_type == 'StokesFlow':
            move_Walkers = DRLPDE_functions.EvaluateWalkers.move_Walkers_Stokes_unsteady
    else:
        if pde_type == 'NavierStokes':
            move_Walkers = DRLPDE_functions.EvaluateWalkers.move_Walkers_NS_steady
        elif pde_type == 'Elliptic':
            move_Walkers = DRLPDE_functions.EvaluateWalkers.move_Walkers_Elliptic
        elif pde_type == 'StokesFlow':
            move_Walkers = DRLPDE_functions.EvaluateWalkers.move_Walkers_Stokes_steady
    
    if pde_type == 'NavierStokes' or 'StokesFlow':
        evaluate_model = DRLPDE_functions.EvaluateWalkers.evaluate_model_NS
    else:
        evaluate_model = DRLPDE_functions.EvaluateWalkers.evaluate_model_PDE
        
        move_walkers_param["drift"] = DRLPDE_param.drift
        eval_model_param["reaction"] = DRLPDE_param.reaction
    
    ### Import functions
    if is_unsteady:
        # Include time range in bounding box
        boundingbox.append(DRLPDE_param.time_range)
        init_con = DRLPDE_param.init_con
    
    ### Organize parameters related to deep learning solver
    num_walkers = DRLPDE_param_solver.num_walkers
    num_ghost = DRLPDE_param_solver.num_ghost
    num_batch = DRLPDE_param_solver.num_batch
    update_walkers_every = 10
    
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
    
    alpha = 1
    update_alpha_every = 100
    update_alpha_tick = 0.1

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

    ### Create Boundary points: BPoints
    ### Organize into DataLoader
    if there_are_boundaries:
        BPoints = DRLPDE_functions.DefineDomain.Boundary_Data(num_bdry, boundingbox, Domain.boundaries, is_unsteady)
        BPoints_batch = torch.utils.data.DataLoader(BPoints, batch_size=num_batch_bdry, shuffle=True)

    if is_unsteady:
        InitPoints = DRLPDE_functions.DefineDomain.Initial_Data(num_init, boundingbox, Domain.boundaries, init_con)
        InitPoints_batch = torch.utils.data.DataLoader(InitPoints, batch_size=num_batch_init, shuffle=True)

    
    ################ Training functions #################

    def train_interior(Batch, model, max_loss, do_resample=False):

        ### Indices to resample
        ### dtype might be unnecessarily big
        resample_index = torch.tensor([], dtype=torch.int64)
        new_max = torch.tensor([0], dtype=torch.float64)

        for Xold, index in Batch:

            # Send to GPU and set requires grad flag
            Xold = Xold.to(dev).requires_grad_(True)

            # Evaluate at old location and Move walkers
            Xnew, Uold, outside = move_Walkers(Xold, model, Domain, **move_walkers_param)

            # Evaluate at new location and average
            Target = evaluate_model(Xold.repeat(num_ghost,1), Xnew, model, **eval_model_param).reshape(num_ghost, 
                                                                                 num_batch,
                                                                                 output_dim).mean(0)

            loss = torch.norm(Target, dim=1)

            if do_resample:
                ### Rejection sampling ###
                # Use the provided max_loss
                #     from previous iteration, always behind by 1
                #     otherwise, would have to redo batches
                # generate uniform( max_loss )
                # Compare sample < uniform
                # Then provide index to resample 
                check_sample = max_loss*torch.rand(X.size(0), device=X.device)
                resample_index = torch.cat( (resample_index, index[loss < check_sample]))

            ### Recalculate max loss
            new_max = torch.max( new_max, torch.sqrt( torch.max(loss).data ))

            loss = torch.mean( loss )

        return loss, new_max, resample_index

    #print("Training has begun")

    ############### First Training Step #####################

    loss, max_loss, resample_index = train_interior(InPoints_batch, model, torch.tensor([100]))

    loss.backward

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

    print('No errors in first epoch')

    ################ Continue Training #########################
    start_time = time.time()

    for step in range(num_epoch):

        do_resample = (step+1) % update_walkers_every == 0 
        loss, max_loss, resample_index = train_interior(InPoints_batch, model, max_loss, do_resample)

        loss.backward

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
        if do_resample and any(resample_index):
            InPoints.location[resample_index,:] = DRLPDE_functions.DefineDomain.generate_interior_points(resample_index.size(0), boundingbox, Domain.boundaries)
            InPoints_Batch = torch.utils.data.DataLoader(InPoints, batch_size=num_batch, shuffle=True)

        # Print statements
        if step == 0:
            current_time = time.time()
            print('Approx time: {:.0f} minutes'.format((current_time - start_time)*num_epoch/60))
        if (step+1) % update_print_every == 0:
            current_time = time.time()
            print('step = {0}/{1}, {2:2.3f} s/step, time-to-go:{3:2.0f} min'.format(
                    step+1, num_epoch, (current_time - start_time) / (step + 1), 
                (current_time - start_time) / (step + 1) * (num_epoch - step - 1)/60))

    # Save model state_dict as pickle file
    if DRLPDE_param.savemodel:
        torch.save(model.state_dict(), "savedmodels/" + DRLPDE_param.savemodel + ".pt")
        #print("model saved as savedmodels/" + DRLPDE_param.savemodel + ".pt")

    # Return the model for plotting
    return model
   
          
if __name__ == "__main__":

    ### Main training step ###
    import argparse
    parser = argparse.ArgumentParser(description="Continuation Method to solve PDEs")
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    