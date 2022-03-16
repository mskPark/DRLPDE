#########   Deep Reinforcement Learning of Partial Differential Equations

#########   Built-in packages

import numpy as np
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim

##########   Main execution

def main_training(param='DRLPDE_param_problem',
                  use_cuda='torch.cuda.is_available()'):
    
    ################# Pre-processing ##################
    ###
    ### Unpack and organize parameters
    
    import DRLPDE_nn
    import DRLPDE_param_solver
    import DRLPDE_functions
    
    import importlib
    
    if param=='DRLPDE_param_problem':
        DRLPDE_param = importlib.import_module("DRLPDE_param_problem")
        print("Pre-processing: Loading parameters from default location: DRLPDE_param_problem.py")
    else:
        DRLPDE_param = importlib.import_module("." + param, package='examples')
        print("Pre-processing: Loading parameters from " + param + '.py')
        
    ### Use cuda
    dev = torch.device("cuda:0" if use_cuda else "cpu")

    ### Unpack and organize variables related to the PDE problem
    
    domain = DRLPDE_param.domain
    my_bdry = DRLPDE_param.my_bdry
    pde_type = DRLPDE_param.pde_type
    is_unsteady = DRLPDE_param.is_unsteady
    output_dim = DRLPDE_param.output_dim
    
    ###
    ### TODO: nn_type in DRLPDE_param_solver to choose the neural network
  
    nn_param = {'depth': DRLPDE_param_solver.nn_depth,
                'width': DRLPDE_param_solver.nn_width,
                'x_dim':DRLPDE_param.x_dim,
                'is_unsteady':DRLPDE_param.is_unsteady ,
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
                       
    if is_unsteady:
        domain.append(DRLPDE_param.time_range)
        init_con = DRLPDE_param.init_con
        
        if pde_type == 'NavierStokes':
            move_Walkers = DRLPDE_functions.move_Walkers_NS_unsteady
        elif pde_type == 'Parabolic':
            move_Walkers = DRLPDE_functions.move_Walkers_Parabolic
    else:
        if pde_type == 'NavierStokes':
            move_Walkers = DRLPDE_functions.move_Walkers_NS_steady
        elif pde_type == 'Elliptic':
            move_Walkers = DRLPDE_functions.move_Walkers_Elliptic
    
    if pde_type == 'NavierStokes':
        evaluate_model = DRLPDE_functions.evaluate_model_NS
    else:
        evaluate_model = DRLPDE_functions.evaluate_model_PDE
        
        move_walkers_param["drift"] = DRLPDE_param.drift
        eval_model_param["reaction"] = DRLPDE_param.reaction
    
    num_walkers = DRLPDE_param_solver.num_walkers
    num_ghost = DRLPDE_param_solver.num_ghost
    num_batch = DRLPDE_param_solver.num_batch

    update_walkers = DRLPDE_param_solver.update_walkers
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
    
    print("Initializing the model")
    
    # Make domain
    boundaries = DRLPDE_functions.make_boundaries(my_bdry)

    if pde_type == 'Navier Stokes':
        MyNeuralNetwork = DRLPDE_nn.IncompressibleNN
    else:
        MyNeuralNetwork = DRLPDE_nn.FeedForwardNN

    # Initialize Model

    if DRLPDE_param.loadmodel:
        model = torch.load("savedmodels/" + DRLPDE_param.loadmodel + ".pt")
        print("Using model from savedmodels/" + DRLPDE_param.loadmodel + ".pt")
    else:
        model = MyNeuralNetwork(**nn_param).to(dev)

    mseloss = nn.MSELoss(reduction = 'mean')
    optimizer = optim.Adam(model.parameters(), 
                           lr=DRLPDE_param_solver.learning_rate, 
                           betas=DRLPDE_param_solver.adam_beta, 
                           weight_decay=DRLPDE_param_solver.weight_decay)

    # Create Walkers and Boundary points 

    RWalkers = DRLPDE_functions.Walker_Data(num_walkers, domain, boundaries)
    RWalkers_batch = torch.utils.data.DataLoader(RWalkers, batch_size=num_batch, shuffle=True)

    if update_walkers == 'move':
        move_RWalkers = torch.zeros_like(RWalkers.location)

    BPoints = DRLPDE_functions.Boundary_Data(num_bdry, domain, boundaries, is_unsteady)
    BPoints_batch = torch.utils.data.DataLoader(BPoints, batch_size=num_batch_bdry, shuffle=True)

    if is_unsteady:
        InitPoints = DRLPDE_functions.Initial_Data(num_init, domain, boundaries, init_con)
        InitPoints_batch = torch.utils.data.DataLoader(InitPoints, batch_size=num_batch_init, shuffle=True)

        
    ################ Training the model #################
    
    print("Training has begun")
    
    start_time = time.time()

    for step in range(num_epoch):

        # Random Walkers - do in batches
        for Xold, index in RWalkers_batch:

            # Send to GPU and set requires grad flag
            Xold = Xold.to(dev).requires_grad_(True)

            # Move walkers
            Xnew, Uold, outside = move_Walkers(Xold, model, boundaries, **move_walkers_param)

            # Evaluate at new location and average
            Unew = evaluate_model(Xold, Xnew, model, **eval_model_param).reshape(num_ghost, num_batch, output_dim).mean(0)
            
            # Calculate loss
            loss = lambda_bell*mseloss(Uold, Unew)
            loss.backward()

            # If moving walkers
            if update_walkers == 'move':
                if any(outside):
                    Xnew[:num_batch,:][outside,:] = DRLPDE_functions.generate_interior_points(torch.sum(outside), 
                                                                                              domain, boundaries)
                move_RWalkers[index,:] = Xnew[:num_batch].detach().cpu()


        # Boundary Points - Do in batches
        for Xbdry, Ubtrue in BPoints_batch:
            Xbdry = Xbdry.to(dev).requires_grad_(True)
            Ubtrue = Ubtrue.to(dev)
            Ubdry = model(Xbdry)
            loss = lambda_bdry*mseloss(Ubdry, Ubtrue)
            loss.backward()

        # Initial Points - Do in batches
        if is_unsteady:
            for Xinit, Uinit_true in InitPoints_batch:
                Xinit = Xinit.to(dev).requires_grad_(True)
                #Uinit_true = Uinit_true.to(dev)
                Uinit = model(Xinit)
                loss = lambda_bdry*mseloss(Uinit, Uinit_true.to(dev))
                loss.backward()

        # Make optimization step
        optimizer.step()
        optimizer.zero_grad()

        # Update walkers

        if (step+1) % update_walkers_every == 0:
            if update_walkers == 'move':
                RWalkers.location = move_RWalkers
                RWalkers_Batch = torch.utils.data.DataLoader(RWalkers, batch_size=num_batch, shuffle=True)
            elif update_walkers == 'remake':
                RWalkers = DRLPDE_functions.Walker_Data(num_walkers, domain, boundaries)
                RWalkers_Batch = torch.utils.data.DataLoader(RWalkers, batch_size=num_batch, shuffle=True)

        # Print statements
        if step == 0:
            print('No errors in first epoch: Training will continue')
        if (step+1) % update_print_every == 0:
            current_time = time.time()
            np.set_printoptions(precision=2)
            print('step = {0}/{1}, {2:3.1f} s/step, time-to-go:{3:2.0f}s'.format(
                    step+1, num_epoch, (current_time - start_time) / (step + 1), 
                (current_time - start_time) / (step + 1) * (num_epoch - step - 1)))

    # Save model 
    if DRLPDE_param.savemodel:
        torch.save(model, "savedmodels/" + DRLPDE_param.savemodel + ".pt")
        print("model saved savedmodels/" + DRLPDE_param.savemodel + ".pt")

    # Return the model and domain for plotting
    return model, domain
   
          
if __name__ == "__main__":

    ### Main training step ###
    import argparse
    parser = argparse.ArgumentParser(description="Starts the Deep Reinforcement Learning of PDEs")
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
        
    model, domain = main_training(param, use_cuda)
          
    ### Save stuff, plot stuff ###
    print("Plotting now")
    
    
    
    
    
    
    
    
    
    
    
    
    