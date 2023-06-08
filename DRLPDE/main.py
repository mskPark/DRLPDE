import torch
import numpy as np
import time
import datetime
import pickle
import importlib

import DRLPDE.create as create
import DRLPDE.neuralnets as neuralnets
import DRLPDE.stochastic as stochastic

### Pytorch default datatype is float32, to change, uncomment the line below

#torch.set_default_dtype(torch.float32)
torch.set_default_dtype(torch.float64)

def define_solver_parameters(**solver):

    ## Default Training parameters
    now = datetime.datetime.now()

    solver_parameters = {'savemodel': now.strftime('%b%d_%I%M%p'),
                         'loadmodel': '',
                         'numpts': 2**12,
                         'numbatch': 2**11,
                         'trainingsteps': 1e4,
                         'neuralnetwork':'FeedForward',
                         'method': {'type':'stochastic',
                                      'dt':1e-4,
                                      'num_ghost':64,
                                      'tol': 1e-6},
                         'learningrate': 1e-4,
                         'bdry_lr': 1e-5,
                         'reschedule_every': 1.1,
                         'resample_every': 1.1,
                         'walk': False,              
                         'importance_sampling': False
                           }
    
    ## 'method': {'type': 'stochastic', 'dt':1e-4, 'num_ghost':128, 'tol':1e-6}
    #          : {'type': 'autodiff'}
    #          : {'type': 'direct}

    ## Additional Parameters to add
    ##

    ### Any new parameters override the default parameters
    for new in solver.keys():
        solver_parameters[new] = solver[new]

    return solver_parameters

def define_problem_parameters(parameters, solver_parameters):

    if parameters:
        param = importlib.import_module("." + parameters, package='examples')
        problem = 'examples/' + parameters + '.py'
    else:
        import DRLPDE.parameters as param
        problem = 'default_parameters.py'

    problem_parameters = {'problem': problem,
                          'input_dim': [param.x_dim, param.t_dim, param.hyper_dim],
                          'output_dim': param.output_dim,
                          'input_range': param.boundingbox + param.t_range + param.hyper_range,
                          'Boundaries': [param.boundingbox,
                                         param.list_of_walls, 
                                         param.solid_walls, 
                                         param.inlet_outlet, 
                                         param.list_of_periodic_ends, 
                                         param.mesh]}

    if param.collect_error:
        problem_parameters['error'] = {'collect_error': param.collect_error,
                                        'num_error': param.num_error,
                                        'true_fun': param.true_fun}
    else:
        problem_parameters['error'] = {'collect_error': param.collect_error}

    if param.t_dim:
        problem_parameters['IC'] = param.init_con

    ### Method type
    if solver_parameters['method']['type'] == 'autodiff':
        import DRLPDE.autodiff as method
        var_train = {'x_dim': param.x_dim,
                    'diffusion': param.diffusion,
                     'forcing': param.forcing}
        
         ### PDE
        if param.t_dim:
            if param.pde_type == 'NavierStokes':
                make_target = method.unsteadyNavierStokes
            elif param.pde_type == 'viscousBurgers':
                make_target = method.unsteadyViscousBurgers
        else:
            if param.pde_type == 'NavierStokes':
                make_target = method.steadyNavierStokes
            elif param.pde_type == 'viscousBurgers':
                make_target = method.steadyViscousBurgers
            elif param.pde_type == 'Stokes':
                make_target = method.Laplace
            elif param.pde_type == 'Laplace':
                make_target = method.Laplace

    elif solver_parameters['method']['type'] == 'stochastic':
        import DRLPDE.stochastic as method
        var_train = {'x_dim': param.x_dim,
                    'diffusion': param.diffusion,
                    'forcing': param.forcing, 
                    'dt': solver_parameters['method']['dt'],
                    'num_ghost': solver_parameters['method']['num_ghost'], 
                    'tol': solver_parameters['method']['tol']
                    }

        ### PDE
        if param.t_dim:
            var_train['ic'] = param.init_con
            if param.pde_type == 'NavierStokes':
                make_target = method.unsteadyNavierStokes
            elif param.pde_type == 'viscousBurgers':
                make_target = method.unsteadyViscousBurgers
            elif param.pde_type == 'Stokes':
                make_target = method.Heat
            elif param.pde_type == 'Heat':
                make_target = method.Heat
        else:
            if param.pde_type == 'NavierStokes':
                make_target = method.steadyNavierStokes
            elif param.pde_type == 'viscousBurgers':
                make_target = method.steadyViscousBurgers
            elif param.pde_type == 'Stokes':
                make_target = method.Laplace
            elif param.pde_type == 'Laplace':
                make_target = method.Laplace

    elif solver_parameters['method']['type'] == 'direct':
        var_train = {'true': param.true_fun,
                     'x_dim': param.x_dim,
                    'diffusion': param.diffusion,
                     'forcing': param.forcing,
                     'dt': 1e-4}
        make_target = create.Dirichlet_target

    elif solver.method == 'finitediff':
        pass
    
    problem_parameters['var_train'] = var_train
    problem_parameters['InteriorTarget'] = make_target

    return problem_parameters

def define_neuralnetwork(problem_parameters, solver_parameters):

    # TODO: More Neural Network Architectures
    if solver_parameters['neuralnetwork'] == 'FeedForward':
        MyNeuralNetwork = neuralnets.FeedForwardNN
    elif solver_parameters['neuralnetwork'] == 'Incompressible':
        MyNeuralNetwork = neuralnets.IncompressibleNN

    nn_size = solver_parameters['nn_size']

    if solver_parameters['loadmodel']:
        model = MyNeuralNetwork(problem_parameters['input_dim'],
                                problem_parameters['output_dim'],
                                **nn_size)
        model.load_state_dict(torch.load("savedmodels/" + solver_parameters['loadmodel'] + ".pt"))
        print("Using model from savedmodels/" + solver_parameters['loadmodel']+ ".pt")
    else:
        model = MyNeuralNetwork(problem_parameters['input_dim'],
                                problem_parameters['output_dim'],
                                **nn_size)

    return model

def solvePDE(parameters='', **solver):

    ### Pytorch default datatype is float32, to change, uncomment the line below
    # torch.set_default_dtype(torch.float64)
    
    # Use GPU if available
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    ### Training Parameters
    solver_parameters = define_solver_parameters(**solver)

    ### Problem Parameters
    problem_parameters = define_problem_parameters(parameters, solver_parameters)

    # TODO Maybe get rid of this
    trainingsteps = int(solver_parameters['trainingsteps'])
    num = solver_parameters['numpts']
    numbatch = solver_parameters['numbatch']

    resample_every = solver_parameters['resample_every']
    walk = solver_parameters['walk']
    importance_sampling = solver_parameters['importance_sampling']
    reschedule_every = solver_parameters['reschedule_every']
    print_every = round(trainingsteps/10)

    ### Neural Network
    model = define_neuralnetwork(problem_parameters, solver_parameters).to(dev)

    ### Create domain and points
    
    # Define the Physical domain through its boundaries
    Domain = create.theDomain(problem_parameters)

    # Create the training points
    Points = create.thePoints(num, Domain, model, problem_parameters, solver_parameters, dev)

    # Squared losses
    # squaredlosses[:, :, 0] = L2 squared loss
    # squaredlosses[:, :, 1] = Linf squared loss

    squaredlosses = np.ones((trainingsteps, Points.numtype, 2))

    # TODO Collect errors
    collect_error = problem_parameters['error']['collect_error']
    if collect_error:
        squarederrors = np.ones((trainingsteps+1, Points.numtype, 2))
        ErrorPoints = create.forError(problem_parameters['error']['num_error'],
                                      Domain,
                                      problem_parameters,
                                      dev)
        squarederrors[0,:,:] = ErrorPoints.CalculateError(model, dev, numbatch)

    # Train once
    squaredlosses[0,:,:] = Points.TrainL2LinfLoss(model, Domain, dev, numbatch, squaredlosses[0,:,:])

    if collect_error:
        squarederrors[1,:,:] = ErrorPoints.CalculateError(model, dev, numbatch)

    # Continue training
    start_time = time.time()
    for step in range(1,trainingsteps):
        
        do_reschedule = step % reschedule_every == 0
        do_resample = step % resample_every == 0
        squaredlosses[step,:,:] = Points.TrainL2LinfLoss(model, Domain, dev, numbatch, squaredlosses[step-1,:,:], importance_sampling)
        
        if collect_error:
            squarederrors[step+1,:,:] = ErrorPoints.CalculateError(model, dev, numbatch)

        if walk:
            Points.toTrain[0].location = stochastic.walk( Points.toTrain[0], num, model, Domain, dev,
                                                            problem_parameters['input_dim'], 
                                                            problem_parameters['input_range'],
                                                            **problem_parameters['var_train'])
        
        if do_reschedule:
            Points.SchedulerStep(squaredlosses[step,:,0])
        
        # Resample points
        if do_resample:
            Points.ResamplePoints(Domain, dev, problem_parameters)

        # Print Progress
        if step % print_every == 0:
            current_time = time.time() - start_time
            print('step = {0} of {1}, Elapsed Time:{2:2.0f} min, Time to Go:{3:2.0f} min'.format(step, trainingsteps, current_time/60, current_time*(trainingsteps - step)/step/60))
    

    # Organize and export losses and errors
    # TODO
    # collect_losses[:,ii] = [ L2loss of region ii, Linfloss of region ii ]
    
    # Save losses and errors
    with open('experiments/' + solver_parameters['savemodel'] + '_losses.pickle', 'wb' ) as handle:
        pickle.dump(squaredlosses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if collect_error:
        with open('experiments/' + solver_parameters['savemodel'] + '_errors.pickle', 'wb' ) as handle:
            pickle.dump(squaredlosses, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save solver parameters
    with open('experiments/' + solver_parameters['savemodel'] + '_parameters.pickle', 'wb') as handle:
        pickle.dump(solver_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save model
    torch.save(model.state_dict(), "savedmodels/" + solver_parameters['savemodel'] + ".pt")
    
    print('Done')

    return model, squaredlosses


if __name__ == "__main__":

    ### Main training step ###
    import argparse
    parser = argparse.ArgumentParser(description="DRLPDE: Deep Reinforcement Learning for solving PDEs")
    parser.add_argument('-example', type=str)
    parser.add_argument('-solver', type=str)
    parser.add_argument('-use_cuda', type=bool)
    parser.add_argument('-dtype', type=str)
    
    args = parser.parse_args()
    
    if args.example:
        param = args.example
    else:
        param = ''

    if args.solver:
        solver = args.solver
    else:
        solver = ''
        
    if args.use_cuda:
        use_cuda = args.use_cuda
    else:
        use_cuda = torch.cuda.is_available()

    if args.dtype:
        dtype = args.dtype
    else:
        dtype = 'single'

    use_model = solvePDE(param, solver, use_cuda, dtype)
