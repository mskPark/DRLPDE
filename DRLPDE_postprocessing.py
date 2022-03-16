###
### Plotting file 
###

import numpy as np
import matplotlib.pyplot as plt
import math
import torch

num_plot_xdim = 128
num_plot_ydim = 128
num_plot_bdry = 64

def plot(param='DRLPDE_param_problem',
             model=''):
    
    ################# Pre-processing ##################
    
    import DRLPDE_nn
    import DRLPDE_functions
    
    ### Unpack variables from DRLPDE_param_problem
    if param=='DRLPDE_param_problem':
        DRLPDE_param = importlib.import_module("DRLPDE_param_problem")
        print("Pre-processing: Loading parameters from default location: DRLPDE_param_problem.py")
    else:
        DRLPDE_param = importlib.import_module("." + param, package='examples')
        print("Pre-processing: Loading parameters from " + param + '.py')
        
    domain = DRLPDE_param.domain
    my_bdry = DRLPDE_param.my_bdry
    output_dim = DRLPDE_param.output_dim
    analytic_sol = DRLPDE_param.exists_analytic_sol
    
    boundaries = DRLPDE_functions.make_boundaries(my_bdry)
    
    num_fig = 1 + model + analytic_sol
    
    ### Plot the domain
    fig_domain, ax_domain = plt.subplots()
    # Plot each boundary
    for bdry in boundaries:
        X_in_bdry =  bdry.plot_bdry(num_plot_bdry, domain, False)[0]

        ax_domain.plot( X_in_bdry[:,0].numpy(), X_in_bdry[:,1].numpy() )

    ### Surface plot for model solution
    if model:
        
        
        Xplot = 
        surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('teste.pdf')
        plt.show()
    ### Surface plot for analytic solution
    if analytic_sol:
        true_sol = DRLPDE_param.true_sol()
    
    ### Error plots TODO: How to calculate errors? Monte Carlo Integration?
    if model and DRLPDE_param.exists_analytic_sol:
        
    
    
if __name__ == "__main__":

    ### Plotting from console ###
    import argparse
    parser = argparse.ArgumentParser(description="Starts the Deep Reinforcement Learning of PDEs")
    parser.add_argument('-example', type=str)
    parser.add_argument('-model', type=str)
    
    args = parser.parse_args()
    
    if args.example:
        param = args.example
    else:
        param = 'DRLPDE_param_problem'
        
    if args.model:
        model = args.model
    else:
        model = ''
        
    plot(param, model)
          