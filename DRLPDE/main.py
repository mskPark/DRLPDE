import torch
import numpy as np
import time
import datetime

import pickle
import importlib

import DRLPDE.create as create
import DRLPDE.train as train
import DRLPDE.diagnostics as diagnostics
import DRLPDE.neuralnets as neuralnets

### Pytorch default datatype is float32, to change, uncomment the line below
# torch.set_default_dtype(torch.float64)

def define_solver_parameters(**solver):

    ## Default Training parameters
    now = datetime.datetime.now()

    solver_parameters = {'savemodel': now.strftime('%b%d_%I%M%p'),
                         'loadmodel': '',
                         'numpts': 2**12,
                         'numbatch': 2**12,
                         'trainingsteps': 5e3,
                         'neuralnetwork':'FeedForward',
                         'nn_size': {'depth':4,
                                       'width':64},
                         'method': {'type':'stochastic',
                                      'dt':1e-4,
                                      'num_ghost':128,
                                      'tol': 1e-6},
                         'optimizer': {'learningrate': 1e-4,
                                         'beta': (0.9, 0.999),
                                         'weightdecay': 0.0},
                         'weights':{'interior':1e0,
                                      'wall':1e0,
                                      'solid':1e0,
                                      'inletoutlet':1e0,
                                      'mesh':1e0,
                                      'ic':1e0},
                         'resample_every': 1.1,
                         'walk': False,              
                         'importance_sampling': False,
                         'adaptive_weighting': { 'reweight_every':1.1,
                                                 'stepsize':1e1},
                         'hybrid': {'num_mesh': 20,
                                    'solvemeshevery': 50}
                           }

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
        problem_parameters['error'] = {'num_error': param.num_error,
                                       'true_fun': param.true_fun}
    else:
        problem_parameters['error'] = {'num_error': 0}

    if param.t_dim:
        problem_parameters['IC'] = {param.init_con}

    ### Method type
    if solver_parameters['method']['type'] == 'autodiff':
        import DRLPDE.autodiff as method
        var_train = {'diffusion': param.diffusion,
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
        var_train = {'diffusion': param.diffusion,
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
        var_train = {'true_fun': param.true_fun}
        make_target = train.Direct_target
        
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
    
    ### Training Parameters
    solver_parameters = define_solver_parameters(**solver)

    ### Problem Parameters
    problem_parameters = define_problem_parameters(parameters, solver_parameters)

    # TODO Get rid of this
    trainingsteps = int(solver_parameters['trainingsteps'])
    num = solver_parameters['numpts']
    numbatch = solver_parameters['numbatch']
    resample_every = solver_parameters['resample_every']
    walk = solver_parameters['walk']
    importance_sampling = solver_parameters['importance_sampling']
    reweight_every = solver_parameters['adaptive_weighting']['reweight_every']
    stepsize = solver_parameters['adaptive_weighting']['stepsize']
    print_every = 1.1 #round(trainingsteps/10)


    ### Neural Network
    model = define_neuralnetwork(problem_parameters, solver_parameters).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=solver_parameters['optimizer']['learningrate'], 
                                 betas=solver_parameters['optimizer']['beta'], 
                                 weight_decay=solver_parameters['optimizer']['weightdecay'])

    # Try LBFGS
    #   Need to figure out closure
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)

    ### Create domain and points
    
    # Define the Physical domain through its boundaries
    Domain = create.theDomain(problem_parameters)

    ### TODO
    ### Incorporate resample/walk inside Points
    Points = create.thePoints(Domain, problem_parameters, solver_parameters)

    # Train once
    for ii in range(len(Points.toTrain)):
        L2loss, Linfloss, resample_index = train.L2Linfloss(Points)
        diagnostics.CollectErrors(Points.forError)

    ###
    

    # Create Interior points and organize into batches
    IntPoints = create.InteriorPoints(num, Domain, input_dim, input_range)
    IntPoints_batch = torch.utils.data.DataLoader(IntPoints, batch_size=numbatch, shuffle=True)
    weight_interior = solver_parameters['weights']['interior']

    # Create Dirichlet Wall points and organize into batches
    if there_are_walls:
        num_wall = create.numBCpoints(num, input_dim, Domain, Domain.wall)
        BCPoints = create.BCPoints(num_wall, Domain, input_dim, input_range)
        BCPoints_batch = torch.utils.data.DataLoader(BCPoints, batch_size=numbatch, shuffle=True)
        weight_wall = solver_parameters['weights']['wall']

    # Create Solid Wall points and organize into batches
    if there_are_solids:
        num_solid = create.numBCpoints(num, input_dim, Domain, Domain.solid)
        SolidPoints = create.SolidWallPoints(num_solid, Domain, input_dim, input_range)
        SolidPoints_batch = torch.utils.data.DataLoader(SolidPoints, batch_size=numbatch, shuffle=True)
        weight_solid = solver_parameters['weights']['solid']

    # Create Inlet/Outlet points and organize into batches

    if there_are_inletoutlets:
        num_inout = create.numBCpoints(num, input_dim, Domain, Domain.inletoutlet)
        InletOutletPoints = create.InletOutletPoints(num_inout, Domain, input_dim, input_range)
        InletOutletPoints_batch = torch.utils.data.DataLoader(InletOutletPoints, batch_size=numbatch, shuffle=True)
        weight_solid = solver_parameters['weights']['inletoutlet']

    # Create Mesh points and organize into batches
    if there_are_meshes:
        num_mesh= solver_parameters['hybrid']['num_mesh']
        solvemesh_every = solver_parameters['hybrid']['solvemesh_every']

        MeshPoints = create.MeshPoints(num_mesh, Domain.mesh[0], model)
        MeshPoints_batch = torch.utils.data.DataLoader(MeshPoints, batch_size=numbatch, shuffle = True)
        weight_mesh = solver_parameters['weights']['mesh']

    # Create Initial Condition points and organize into batches
    if unsteady:
        num_ic = create.numICpoints(num, input_dim)
        ICPoints = create.ICPoints(num_ic, input_dim, input_range, Domain, param.init_con)
        ICPoints_batch = torch.utils.data.DataLoader(ICPoints, batch_size=numbatch, shuffle=True)
        weight_ic = solver_parameters['weights']['ic']

    # Error Diagnostics
    # Only used when true function is given
    if collect_error:
        ### Random points for calculating L2 error through Monte Carlo Integration
        ErrorPoints = create.InteriorPoints(num_error, Domain, input_dim, input_range)
        ErrorPoints_batch = torch.utils.data.DataLoader(ErrorPoints, batch_size=numbatch, shuffle = True)

    ###
    ### Train once
    ###

    optimizer.zero_grad()

    # Interior

    L2loss_interior, Linfloss_interior, resample_interior = train.interior(IntPoints, num, model, make_target, var_train, dev, Domain.volume, weight_interior, 0.0, False)
    
    Total_L2loss_interior = L2loss_interior.cpu().numpy()*np.ones(trainingsteps)
    Total_Linfloss_interior = Linfloss_interior.cpu().numpy()*np.ones(trainingsteps)

    # Walls
    if there_are_walls:
        L2loss_wall, Linfloss_wall, resample_wall = train.boundary(BCPoints_batch, num_wall, model, train.Dirichlet_target, dev, weight_wall, 0.0, False)
        
        Total_L2loss_wall = L2loss_wall.cpu().numpy()*np.ones(trainingsteps)
        Total_Linfloss_wall = Linfloss_wall.cpu().numpy()*np.ones(trainingsteps)

    if there_are_solids:
        L2loss_solid, Linfloss_solid, resample_solid = train.boundary(SolidPoints_batch, num_solid, model, train.Dirichlet_target, dev, weight_solid, 0.0, False)

        Total_L2loss_solid = L2loss_solid.cpu().numpy()*np.ones(trainingsteps)
        Total_Linfloss_solid = Linfloss_solid.cpu().numpy()*np.ones(trainingsteps)

    if there_are_inletoutlets:
        L2loss_inletoutlet, Linfloss_inletoutlet, resample_inletoutlet = train.boundary(InletOutletPoints_batch, num_inout, model, train.Inletoutlet_target, dev, weight_inletoutlet, 0.0, False)
    
        Total_L2loss_inletoutlet = L2loss_inletoutlet.cpu().numpy()*np.ones(trainingsteps)
        Total_Linfloss_inletoutlet = Linfloss_inletoutlet.cpu().numpy()*np.ones(trainingsteps)

    if there_are_meshes:
        L2loss_mesh, Linfloss_mesh, [] = train.boundary(MeshPoints_batch, num_mesh, model, train.Dirichlet_target, dev, weight_mesh, 0.0, False)

        Total_L2loss_mesh = L2loss_mesh.cpu().numpy()*np.ones(trainingsteps)
        Total_Linfloss_mesh = Linfloss_mesh.cpu().numpy()*np.ones(trainingsteps)

    if unsteady:
        L2loss_ic, Linfloss_ic, resample_ic = train.boundary(ICPoints_batch, num_ic, model, train.Dirichlet_target, dev, weight_ic, 0.0, False)

        Total_L2loss_ic = L2loss_ic.cpu().numpy()*np.ones(trainingsteps)
        Total_Linfloss_ic = Linfloss_ic.cpu().numpy()*np.ones(trainingsteps)

    if collect_error:
        ErrorPoints.location = create.generate_interior_points(num_error, input_dim, input_range, Domain, Domain.inside)
        L2error, Linferror = diagnostics.CalculateError(ErrorPoints_batch, num_error, Domain.volume, model, true_fun, dev)

        Total_L2error = L2error.cpu().numpy()*np.ones(trainingsteps)
        Total_Linferror = Linferror.cpu().numpy()*np.ones(trainingsteps)

    optimizer.step()

    print('No errors in first epoch, training will continue')
    
    ###
    ### Continue training
    ###

    start_time = time.time()

    for step in range(trainingsteps-1):
        
        do_resample = (step+1) % resample_every == 0
        do_reweight = (step+1) % reweight_every == 0

        optimizer.zero_grad()

        # Interior
        if do_reweight:
            weight_interior = train.reweight(Linfloss_interior, stepsize)
        L2loss_interior, Linfloss_interior, resample_interior = train.interior(IntPoints_batch, num, model, make_target, var_train, dev, weight_interior, Linfloss_interior, importance_sampling)

        Total_L2loss_interior[step+1] = L2loss_interior.cpu().numpy()
        Total_Linfloss_interior[step+1] = Linfloss_interior.cpu().numpy()

        # Walls
        if there_are_walls:
            if do_reweight:
                weight_wall = train.reweight(Linfloss_wall, stepsize)
            L2loss_wall, Linfloss_wall, resample_wall = train.boundary(BCPoints_batch, num_wall, model, train.Dirichlet_target, dev, weight_wall, Linfloss_wall, importance_sampling)
            
            Total_L2loss_wall[step+1] = L2loss_wall.cpu().numpy()
            Total_Linfloss_wall[step+1] = Linfloss_wall.cpu().numpy()

        if there_are_solids:
            if do_reweight:
                weight_solid = train.reweight(Linfloss_solid, stepsize)
            L2loss_solid, Linfloss_solid, resample_solid = train.boundary(SolidPoints_batch, num_solid, model, train.Dirichlet_target, dev, weight_solid, Linfloss_solid, importance_sampling)

            Total_L2loss_solid[step+1] = L2loss_solid.cpu().numpy()
            Total_Linfloss_solid[step+1] = Linfloss_solid.cpu().numpy()

        if there_are_inletoutlets:
            if do_reweight:
                weight_inletoutlet = train.reweight(Linfloss_inletoutlet, stepsize)
            L2loss_inletoutlet, Linfloss_inletoutlet, resample_inletoutlet = train.boundary(InletOutletPoints_batch, num_inout, model, train.Inletoutlet_target, dev, weight_inletoutlet, Linfloss_inletoutlet, importance_sampling)
            
            Total_L2loss_inletoutlet[step+1] = L2loss_inletoutlet.cpu().numpy()
            Total_Linfloss_inletoutlet[step+1] = Linfloss_inletoutlet.cpu().numpy()

        if there_are_meshes:
            if do_reweight:
                weight_mesh = train.reweight(Linfloss_mesh, stepsize)
            L2loss_mesh, Linfloss_mesh, [] = train.boundary(MeshPoints_batch, num_mesh, model, train.Dirichlet_target, dev, weight_mesh, Linfloss_mesh, False)

            Total_L2loss_mesh[step+1] = L2loss_mesh.cpu().numpy()
            Total_Linfloss_mesh[step+1] = Linfloss_mesh.cpu().numpy()

        if unsteady:
            if do_reweight:
                weight_ic = train.reweight(Linfloss_ic, stepsize)
            L2loss_ic, Linfloss_ic, resample_ic = train.boundary(ICPoints_batch, num_ic, model, train.Dirichlet_target, dev, weight_ic, Linfloss_ic, importance_sampling)

            Total_L2loss_ic[step+1] = L2loss_ic.cpu().numpy()
            Total_Linfloss_ic[step+1] = Linfloss_ic.cpu().numpy()

        if collect_error:
            ErrorPoints.location = create.generate_interior_points(num_error, input_dim, input_range, Domain, Domain.inside)
            L2error, Linferror = diagnostics.CalculateError(ErrorPoints_batch, num_error, Domain.volume, model, true_fun,dev)

            Total_L2error[step+1] = L2error.cpu().numpy()
            Total_Linferror[step+1] = Linferror.cpu().numpy()

        optimizer.step()

        # Resample points

        if do_resample:
            indices = torch.arange(num)
            if any(resample_interior):
                # Only points found in importance sampling routine are being resampled
                indices = indices[resample_interior]
            if walk:
                IntPoints.location[indices,:] = method.walk(IntPoints.location[indices,:], num, model, input_dim, input_range, **var_train)
            else:
                IntPoints.location[indices,:] = create.generate_interior_points(indices.size(0), input_dim, input_range, Domain, Domain.inside)

            if there_are_walls:
                indices = torch.arange(num_wall)
                if any(resample_wall):
                    indices = indices[resample_wall]
                BCPoints.location[indices,:], BCPoints.value[indices,:] = create.generate_boundary_points(indices.size(0), input_dim, input_range, Domain.wall)
            
            if there_are_solids:
                indices = torch.arange(num_solid)
                if any(resample_solid):
                    indices = indices[resample_solid]
                SolidPoints.location[indices,:], SolidPoints.value[indices,:] = create.generate_boundary_points(indices.size(0), input_dim, input_range, Domain.solid)

            if there_are_inletoutlets:
                indices = torch.arange(num_inout)
                if any(resample_inletoutlet):
                    indices = indices[resample_inletoutlet]
                InletOutletPoints.location[indices,:], InletOutletPoints.value[indices,:] = create.generate_boundary_points(indices.size(0), input_dim, input_range, Domain.inletoutlet)

            if unsteady:
                indices = torch.arange(num_ic)
                if any(resample_ic):
                    indices = indices[resample_ic]
                ICPoints.location[indices,:] = create.generate_initial_points(indices.size(0), input_dim, input_range, Domain)
                ICPoints.value[indices,:] = param.init_con(ICPoints.location[indices,:])

        if there_are_meshes:
            do_meshsolve = (step+1) % solvemesh_every == 0
            if do_meshsolve:
                b = Domain.mesh[0].rhs(num_mesh, model)
                MeshPoints.value = MeshPoints.solveU(b)
                    
        # Print statements
        if step == 0:
            current_time = time.time()
            print('Approx time: {:.0f} minutes'.format((current_time - start_time)*trainingsteps/60))

        if (step+1) % print_every == 0:
            current_time = time.time()
            print('step = {0}/{1}, {2:2.3f} s/step, time-to-go:{3:2.0f} min'.format(
                    step+1, trainingsteps, (current_time - start_time) / (step + 1), 
                (current_time - start_time) / (step + 1) * (trainingsteps - step - 1)/60))
    

    ### Collect final loss and errors
    ### Save model
    ### Save pickle file

    ### Final errors
    loss_error = {'loss' : {}, 'errors': {}}
    loss_error['loss']['L2 interior loss'] = Total_L2loss_interior
    loss_error['loss']['Linf interior loss'] = Total_Linfloss_interior

    if there_are_walls:
        loss_error['loss']['L2 wall loss'] = Total_L2loss_wall
        loss_error['loss']['Linf wall loss'] = Total_Linfloss_wall
    if there_are_solids:
        loss_error['loss']['L2 solid loss'] = Total_L2loss_solid
        loss_error['loss']['Linf solid loss'] = Total_Linfloss_solid
    if there_are_inletoutlets:
        loss_error['loss']['L2 inletoutlet loss'] = Total_L2loss_inletoutlet
        loss_error['loss']['Linf inletoutlet loss'] = Total_Linfloss_inletoutlet
    if there_are_meshes:
        loss_error['loss']['L2 mesh loss'] = Total_L2loss_mesh
        loss_error['loss']['Linf mesh loss'] = Total_Linfloss_mesh
    if unsteady:
        loss_error['loss']['L2 ic loss'] = Total_L2loss_ic
        loss_error['loss']['Linf ic loss'] = Total_Linfloss_ic
    if collect_error:
        loss_error['errors']['L2 error'] = Total_L2error
        loss_error['errors']['Linf error'] = Total_Linferror
    
    ### Save as pickle file

    with open('experiments/' + solver_parameters['savemodel'] + '_losserror.pickle', 'wb' ) as handle:
        pickle.dump(loss_error, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('experiments/' + solver_parameters['savemodel'] + '_parameters.pickle', 'wb') as handle:
        pickle.dump(solver_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save model
    torch.save(model.state_dict(), "savedmodels/" + solver_parameters['savemodel'] + ".pt")
    print('Done')

    return model, loss_error


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
