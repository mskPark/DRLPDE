import torch
import math
import numpy as np

import datetime

import DRLPDE.create as create
import DRLPDE.neuralnets as neuralnets

from torchviz import make_dot

import matplotlib.pyplot as plt
import cv2
import imageio

import matplotlib as mpl

mpl.rcParams['figure.dpi']= 300
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 12

### Make contour for scalar
###    Only plot in the domain
###    Make comparison with true solution if available
###    

### Make video for fluids
###    Only plot in the domain
###    Have tracer points



def define_solver_parameters(**solver):

    ## Default Training parameters
    now = datetime.datetime.now()

    solver_parameters = {'savemodel': now.strftime('%b%d_%I%M%p'),
                         'loadmodel': '',
                         'numpts': 2**13,
                         'numbatch': 2**11,
                         'trainingsteps': 1e4,
                         'neuralnetwork':'FeedForward',
                         'nn_size':{'depth':4,
                                    'width':64},
                         'method': {'type':'stochastic',
                                      'dt':1e-4,
                                      'num_ghost':64,
                                      'tol': 1e-6},
                         'learningrate': 1e-3,
                         'interior_weight':1e0,
                         'bdry_weight': 1e-1,
                         'reschedule_every': 1.1,
                         'resample_every': 1.1,
                         'walk': True,              
                         'importance_sampling': False,
                         'collect_losses': 1.1
                           }
    
    ## 'method': {'type': 'stochastic', 'dt':1e-3, 'num_ghost':64, 'tol':1e-6}
    #          : {'type': 'autodiff'}
    #          : {'type': 'direct'}

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
    elif solver_parameters['neuralnetwork'] == 'ResNet':
        MyNeuralNetwork = neuralnets.ResNetNN
    elif solver_parameters['neuralnetwork'] == 'ResNetIncompressible':
        MyNeuralNetwork = neuralnets.ResNetIncompressible

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


def GridPoints(problem_parameters, model):
    ### Return X, Y, Z that will be used in tricontourf
    ### Meshgrid -> remove any outside -> Add boundary points

    

    return X, Y, Z

def scalarcontour(problem_parameters, model):

    Domain = create.theDomain(problem_parameters)
    Points = create.PlotPoints(Domain)

    nn_size = parameters['nn_size']

    model = MyNeuralNetwork(input_dim, output_dim, **nn_size)
    model.load_state_dict(torch.load("../savedmodels/" + test + ".pt"))

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[6.4, 5.0])

    plt.tight_layout(pad=0.75)

    contour = ax[0,0].contourf( x1, x2, u, levels, cmap=plt.cm.coolwarm)

    ax[0,0].set_title(r'$\tilde{u}$')

    ax[0,0].tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )

    colorbar0_param = fig.add_axes(
        [ax[0,0].get_position().x1 + 0.01,
        ax[0,0].get_position().y0,
        0.01,
        ax[0,0].get_position().height])
    colorbar0 = plt.colorbar(contour0, ticks=level_tick, cax = colorbar0_param)
    colorbar0.ax.tick_params(labelsize=10)

    pass

def computationalgraph(model):
    ## Example of using torch viz to show the computational graph

    x = torch.randn((29,2))

    make_dot(model(x), params = dict(model.named_parameters()))
    
    pass

def RK4(xt, u):
    ### Calculate x(t+1) from x' = u(x,t) using Runge-Kutta 4
    ### xt = (x,t)
    ### Used for plotting streamlines and making animations with tracer particles

    xt0 = xt

    h = 0.01

    k1 = u(xt)

    xt[:,:-1] = xt0[:,:-1] + h*k1/2 
    xt[:,-1] = xt0[:,-1] + h/2
    k2 = u(xt)

    xt[:,:-1] = xt0[:,:-1] + h*k2/2 
    k3 = u(xt)

    xt[:,:-1] = xt0[:,:-1] + h*k3
    xt[:,-1] = xt0[:,-1] + h
    k4 = u(xt)

    xt1 = xt0 + h/6 * (k1 + k2 + k3 + k4)
    return xt1

def make_video():

    plt.savefig('foo.png')
    
    img_array = []
    for filename in ('C:/New folder/Images/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter('fancyname.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()