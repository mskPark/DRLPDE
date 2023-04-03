import torch
import numpy as np
import time

import DRLPDE.create as create
import DRLPDE.train as train
import DRLPDE.parameters_solver as solver
import DRLPDE.neuralnets as neuralnets
import importlib

def solvePDE(parameters='',
             use_cuda=torch.cuda.is_available(),
             dtype='single'):
    
    ###
    ### Import and Organization
    ###

    # Pytorch default datatype is float32
    if dtype=='double':
        torch.set_default_dtype(torch.float64)
    
    # Import parameters
    if parameters:
        param = importlib.import_module("." + parameters, package='examples')
    else:
        import DRLPDE.parameters as param
        
    # Use cuda
    dev = torch.device("cuda:0" if use_cuda else "cpu")

    # Dimensions of problem
    input_dim = [param.x_dim, param.t_dim, param.hyper_dim]
    output_dim = param.output_dim

    # Intervals
    input_range = param.boundingbox + param.t_range + param.hyper_range
    
    # Boolean for each wall type
    there_are_walls = any(param.list_of_walls)
    there_are_solids = any(param.solid_walls)
    there_are_inletoutlets = any(param.inlet_outlet)
    there_are_meshes = any(param.mesh)
    unsteady = bool(param.t_dim)

    # Training parameters
    trainingsteps = solver.trainingsteps
    
    num = solver.numpts
    numbatch = solver.numbatch

    resample_every = solver.resample_every
    reweight_every = solver.reweight_every

    print_every = solver.print_every

    if there_are_meshes:
        num_mesh = solver.num_mesh
        solvemesh_every = solver.solvemesh_every

    ###
    ### Initialize the Neural Network
    ###

    # TODO: Choose other neural network architectures from neuralnets.py
    MyNeuralNetwork = neuralnets.IncompressibleNN

    var_nn = {'depth': solver.nn_depth,
              'width': solver.nn_width,
              'input_dim': input_dim,
              'output_dim': output_dim
            }

    if param.loadmodel:
        model = MyNeuralNetwork(**var_nn).to(dev)
        model.load_state_dict(torch.load("savedmodels/" + param.loadmodel + ".pt"))
        print("Using model from savedmodels/" + param.loadmodel + ".pt")
    else:
        model = MyNeuralNetwork(**var_nn).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=solver.learningrate, 
                                 betas=solver.adambeta, 
                                 weight_decay=solver.weightdecay)

    ###
    ### Create domain and points
    ###
    
    # Define the Physical domain through its boundaries
    Domain = create.SpaceDomain(param.boundingbox, 
                                param.list_of_walls, 
                                param.solid_walls, 
                                param.inlet_outlet, 
                                param.list_of_periodic_ends, 
                                param.mesh,
                                param.init_con)

    # Create Interior points and organize into batches
    IntPoints = create.InteriorPoints(num, Domain, input_dim, input_range)
    IntPoints_batch = torch.utils.data.DataLoader(IntPoints, batch_size=numbatch, shuffle=True)

    # Create Dirichlet Wall points and organize into batches
    if there_are_walls:
        num_wall = create.numBCpoints(num, input_dim, Domain, Domain.wall)
        BCPoints = create.BCPoints(num_wall, Domain, input_dim, input_range)
        BCPoints_batch = torch.utils.data.DataLoader(BCPoints, batch_size=numbatch, shuffle=True)

    # Create Solid Wall points and organize into batches
    if there_are_solids:
        num_solid = create.numBCpoints(num, input_dim, Domain, Domain.solid)
        SolidPoints = create.SolidWallPoints(num_solid, Domain, input_dim, input_range)
        SolidPoints_batch = torch.utils.data.DataLoader(SolidPoints, batch_size=numbatch, shuffle=True)

    # Create Inlet/Outlet points and organize into batches
    if there_are_inletoutlets:
        num_inout = create.numBCpoints(num, input_dim, Domain, Domain.inletoutlet)
        InletOutletPoints = create.InletOutletPoints(num_inout, Domain, input_dim, input_range)
        InletOutletPoints_batch = torch.utils.data.DataLoader(InletOutletPoints, batch_size=numbatch, shuffle=True)

    # Create Mesh points and organize into batches
    if there_are_meshes:
        MeshPoints = create.MeshPoints(num_mesh, Domain.mesh[0], model)
        MeshPoints_batch = torch.utils.data.DataLoader(MeshPoints, batch_size=numbatch, shuffle = True)

    # Create Initial Condition points and organize into batches
    if unsteady:
        num_ic = create.numICpoints(num, input_dim)
        ICPoints = create.ICPoints(num_ic, input_dim, input_range, Domain, param.init_con)
        ICPoints_batch = torch.utils.data.DataLoader(ICPoints, batch_size=numbatch, shuffle=True)

    # Create points for calculating error
    #ErrorPoints = create.ErrorPoints(num_error, Domain, input_dim, input_range)
    #ErrorPoints_batch = torch.utils.data.DataLoader(ErrorPoints, batch_size=numbatch, shuffle = True)

    ### Method type
    if solver.method == 'autodiff':
        import DRLPDE.autodiff as method
        var_train = {'diffusion': param.diffusion,
                     'forcing': param.forcing,
                     'x_dim': input_dim[0]
                    }

    elif solver.method == 'stochastic':
        import DRLPDE.stochastic as method
        var_train = {'diffusion': param.diffusion,
                    'forcing': param.forcing, 
                    'x_dim': input_dim[0],
                    'domain': Domain,
                    'dt': solver.dt,
                    'num_ghost': solver.num_ghost, 
                    'tol': solver.tol
                    }
        
    elif solver.method == 'finitediff':
        pass
        # import finitediff as method
        # var_train = {}

    ### PDE
    if unsteady:
        if param.pde_type == 'NavierStokes':
            make_target = method.unsteadyNavierStokes
        elif param.pde_type == 'viscousBurgers':
            make_target = method.unsteadyViscousBurgers
    else:
        if param.pde_type == 'NavierStokes':
            make_target = method.steadyNavierStokes
        elif param.pde_type == 'viscousBurgers':
            make_target = method.steadyViscousBurgers
        elif param.pde_type == 'Laplace':
            make_target = method.Laplace

    if there_are_inletoutlets:
        pass
        #
        # model(X)
        #

    ###
    ### Training the neural network
    ###

    do_resample = False
    Linfloss_interior = 0.0
    Linfloss_wall = 0.0
    Linfloss_solid = 0.0
    Linfloss_inletoutlet = 0.0
    Linfloss_mesh = 0.0
    Linfloss_ic = 0.0

    weight_interior = 1.0
    weight_wall = 1.0
    weight_solid = 1.0
    weight_inletoutlet = 1.0
    weight_mesh = 1.0
    weight_ic = 1.0

    ###
    ### Train once
    ###

    optimizer.zero_grad()

    # Interior

    L2loss_interior, Linfloss_interior, resample_interior = train.interior(IntPoints_batch, num, model, make_target, var_train, dev, weight_interior, Linfloss_interior, False)

    # Walls
    if there_are_walls:
        L2loss_wall, Linfloss_wall, resample_wall = train.boundary(BCPoints_batch, num_wall, model, train.Dirichlet_target, dev, weight_wall, Linfloss_wall, False)

    if there_are_solids:
        L2loss_solid, Linfloss_solid, resample_solid = train.boundary(SolidPoints_batch, num_solid, model, train.Dirichlet_target, dev, weight_solid, Linfloss_solid, False)

    if there_are_inletoutlets:
        L2loss_inletoutlet, Linfloss_inletoutlet, resample_inletoutlet = train.boundary(InletOutletPoints_batch, num_inout, model, train.Inletoutlet_target, dev, weight_inletoutlet, Linfloss_inletoutlet, False)
    
    if there_are_meshes:
        L2loss_mesh, Linfloss_mesh, [] = train.boundary(MeshPoints_batch, num_mesh, model, train.Dirichlet_target, dev, weight_mesh, Linfloss_mesh, False)

    if unsteady:
        L2loss_ic, Linfloss_ic, resample_ic = train.boundary(ICPoints_batch, num_ic, model, train.Dirichlet_target, dev, weight_ic, Linfloss_ic, False)

    # Calculate L2 and Linf error
    #L2error, Linferror = train.L2error(ErrorPoints_batch, num, model, param.bdry_con)
    #Total_L2error = L2error.numpy()*np.ones(trainingsteps+1)
    #Total_Linferror = Linferror.numpy()*np.ones(trainingsteps+1)

    optimizer.step()

    print('No errors in first epoch, training will continue')
    
    ###
    ### Continue training
    ###

    start_time = time.time()

    for step in range(trainingsteps):
        
        optimizer.zero_grad()

        # Interior

        L2loss_interior, Linfloss_interior, resample_interior = train.interior(IntPoints_batch, num, model, make_target, var_train, dev, Linfloss_interior, do_resample)

        do_reweight = (step+1) % reweight_every == 0
        # Walls
        if there_are_walls:
            if do_reweight:
                weight_wall = train.reweight(L2loss_interior, L2loss_wall)
            L2loss_wall, Linfloss_wall, resample_wall = train.boundary(BCPoints_batch, num_wall, model, train.Dirichlet_target, dev, weight_wall, Linfloss_wall, do_resample)
        
        if there_are_solids:
            if do_reweight:
                weight_solid = train.reweight(L2loss_interior, L2loss_solid)
            L2loss_solid, Linfloss_solid, resample_solid = train.boundary(SolidPoints_batch, num_solid, model, train.Dirichlet_target, dev, weight_solid, Linfloss_solid, do_resample)

        if there_are_inletoutlets:
            if do_reweight:
                weight_inletoutlet = train.reweight(L2loss_interior, L2loss_inletoutlet)
            L2loss_inletoutlet, Linfloss_inletoutlet, resample_inletoutlet = train.boundary(InletOutletPoints_batch, num_inout, model, train.Inletoutlet_target, dev, weight_inletoutlet, Linfloss_inletoutlet, do_resample)

        if there_are_meshes:
            L2loss_mesh, Linfloss_mesh, [] = train.boundary(MeshPoints_batch, num_mesh, model, train.Dirichlet_target, dev, weight_mesh, Linfloss_mesh, False)

        if unsteady:
            if do_reweight:
                weight_ic = train.reweight(L2loss_interior, L2loss_ic)
            L2loss_ic, Linfloss_ic, resample_ic = train.boundary(ICPoints_batch, num_ic, model, train.Dirichlet_target, dev, weight_ic, Linfloss_ic, do_resample)

        optimizer.step()

        # TODO Collect diagnostics data: 
        # L2loss, Linfloss
        #   interior, walls, solids, inletoutlets, ic

        # Move walkers (?)
        # TODO

        # Resample points
        do_resample = (step+1) % resample_every == 0
        if do_resample:
            if any(resample_interior):
                IntPoints.location[resample_interior,:] = create.generate_interior_points(torch.sum(resample_interior), input_dim, input_range, Domain)

            if there_are_walls:
                if any(resample_wall):
                    BCPoints.location[resample_wall,:], BCPoints.value[resample_wall,:] = create.generate_boundary_points(torch.sum(resample_wall), input_dim, input_range, Domain.wall)
            if there_are_solids:
                if any(resample_solid):
                    SolidPoints.location[resample_solid,:], SolidPoints.value[resample_solid,:] = create.generate_boundary_points(torch.sum(resample_solid), input_dim, input_range, Domain.solid)

            if there_are_inletoutlets:
                if any(resample_inletoutlet):
                    InletOutletPoints.location[resample_inletoutlet,:], InletOutletPoints.value[resample_inletoutlet,:] = create.generate_boundary_points(torch.sum(resample_inletoutlet), input_dim, input_range, Domain.inletoutlet)

            if unsteady:
                if any(resample_ic):
                    ICPoints.location[resample_ic,:] = create.generate_initial_points(torch.sum(resample_ic), input_dim, input_range, Domain)
                    ICPoints.value[resample_ic,:] = param.init_con(ICPoints.location[resample_ic,:])

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

    # Save model
    if param.savemodel:
        torch.save(model.state_dict(), "savedmodels/" + param.savemodel + ".pt")

        # TODO save details of the neural network
        print('Saved the model state_dict at savedmodels/' + param.savemodel + 'pt' )

    return model

if __name__ == "__main__":

    ### Main training step ###
    import argparse
    parser = argparse.ArgumentParser(description="DRLPDE: Deep Reinforcement Learning for solving PDEs")
    parser.add_argument('-example', type=str)
    parser.add_argument('-use_cuda', type=bool)
    
    args = parser.parse_args()
    
    if args.example:
        param = args.example
    else:
        param = ''
        
    if args.use_cuda:
        use_cuda = args.use_cuda
    else:
        use_cuda = torch.cuda.is_available()

    use_model = solvePDE(param, use_cuda)
