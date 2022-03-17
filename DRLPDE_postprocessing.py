###
### Plotting file 
###

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import torch

num_plot_xdim = 128
num_plot_ydim = 128
num_plot_bdry = 64
num_plot_rand = 256

def plot(param='DRLPDE_param_problem',
         use_model=''):
    
    print("Plotting now")
    
    ################# Pre-processing ##################
    
    import importlib
    import DRLPDE_nn
    import DRLPDE_functions
    
    ### Unpack variables
    if param=='DRLPDE_param_problem':
        DRLPDE_param = importlib.import_module("DRLPDE_param_problem")
        print("Loading parameters from default location: DRLPDE_param_problem.py")
    else:
        DRLPDE_param = importlib.import_module("." + param, package='examples')
        print("Loading parameters from " + param + '.py')
        
    domain = DRLPDE_param.domain
    my_bdry = DRLPDE_param.my_bdry
    output_dim = DRLPDE_param.output_dim
    exists_analytic_sol = DRLPDE_param.exists_analytic_sol
    
    boundaries = DRLPDE_functions.make_boundaries(my_bdry)
    
    ### Plot the domain
    fig_domain, ax_domain = plt.subplots()

    Xbdry = []
    for bdry in boundaries:
        X_each_bdry = bdry.plot_bdry(num_plot_bdry)
        Xbdry.append( X_each_bdry )
        ax_domain.plot( X_each_bdry[:,0].numpy(), X_each_bdry[:,1].numpy(), 'k-' )
    
    fig_domain.savefig("plots/" + param + "_domain.png")
    print("Domain plot saved in " + "plots/" + param + "_domain.png")
    
    ### Plot the solution(s)    
    ### Two options to decide upon
    ### 1. tricontourf, does not need meshgrid
    ### 2. contourf
    
    ### TODO: include levels
        
    method = 1  
        
    # Surface plot for model solution
    if use_model:
        
        model = torch.load("savedmodels/" + use_model + ".pt").to('cpu')
        
        ### Plot boundaries first
        fig_model, ax_model = plt.subplots(nrows=output_dim, ncols=1, figsize=[6.4, output_dim*5.0])
        
        for ii in range(output_dim):
            for jj in range(len(Xbdry)):
                ax_model[ii].plot( Xbdry[jj][:,0].numpy(), Xbdry[jj][:,1].numpy(), 'k-' )
        
        ### TODO: include levels
        
        if method == 1:
            ### Method 1
            Xplot = torch.cat( (torch.cat( Xbdry, dim=0), 
                                DRLPDE_functions.generate_interior_points(num_plot_rand, domain, boundaries)), dim=0 )
            Uplot = torch.split( model(Xplot), 1, dim=1 )
            
            for ii in range(output_dim):
                surf = ax_model[ii].tricontourf(Xplot[:,0].detach().numpy(), 
                                                Xplot[:,1].detach().numpy(), 
                                                Uplot[ii][:,0].detach().numpy(), 
                                                cmap=cm.coolwarm)

        if method == 2:
            ### Method 2
            Xmesh = torch.meshgrid( torch.linspace(domain[0][0], domain[0][1], num_plot_xdim ),
                                    torch.linspace(domain[1][0], domain[1][1], num_plot_ydim ) )
    
            Uplot = model( torch.stack( (Xmesh[0].reshape(-1), 
                                         Xmesh[1].reshape(-1)), dim=1)).reshape(num_plot_xdim, num_plot_ydim, output_dim)

            for ii in range(output_dim):
                surf = ax_model[ii].contourf(Xmesh[0].detach().numpy(), 
                                             Xmesh[1].detach().numpy(), 
                                             Uplot[:,:,ii].detach().numpy(), 
                                             cmap=cm.coolwarm)
            
        fig_model.savefig("plots/" + param + "_model.png")
        print("Model solution plot saved in " + "plots/" + param + "_model.png")
        
    # Surface plot for analytic solution
    if exists_analytic_sol:
        
        true_sol = DRLPDE_param.true_sol
        
        ### Plot boundaries first
        fig_analytic, ax_analytic = plt.subplots(nrows=output_dim, ncols=1, figsize=[6.4, output_dim*5.0])
        
        for ii in range(output_dim):
            for jj in range(len(Xplot)):
                ax_analytic[ii].plot( Xbdry[jj][:,0].numpy(), Xbdry[jj][:,1].numpy(), 'k-' )
        
        ### TODO: include levels
        
        if method == 1:
            ### Method 1
            Xplot = torch.cat( (torch.cat( Xbdry, dim=0), 
                                DRLPDE_functions.generate_interior_points(num_plot_rand, domain, boundaries)), dim=0 )
            Uplot = torch.split( true_sol(Xplot), 1, dim=1 )
            
            for ii in range(output_dim):
                surf = ax_analytic[ii].tricontourf(Xplot[:,0].detach().numpy(), 
                                                Xplot[:,1].detach().numpy(), 
                                                Uplot[ii][:,0].detach().numpy(), 
                                                cmap=cm.coolwarm)

        if method == 2:
            ### Method 2
            Xmesh = torch.meshgrid( torch.linspace(domain[0][0], domain[0][1], num_plot_xdim ),
                                    torch.linspace(domain[1][0], domain[1][1], num_plot_ydim ) )
    
            Uplot = true_sol( torch.stack( (Xmesh[0].reshape(-1), 
                                         Xmesh[1].reshape(-1)), dim=1)).reshape(num_plot_xdim, num_plot_ydim, output_dim)

            for ii in range(output_dim):
                surf = ax_analytic[ii].contourf(Xmesh[0].detach().numpy(), 
                                             Xmesh[1].detach().numpy(), 
                                             Uplot[:,:,ii].detach().numpy(), 
                                             cmap=cm.coolwarm)
        
        fig_analytic.savefig("plots/" + param + "_analytic.png")
        print("Analytic solution plot saved in " + "plots/" + param + "_analytic.png")
        
    ### Error plots TODO: How to calculate errors? Monte Carlo Integration?
    if model and DRLPDE_param.exists_analytic_sol:
        pass
    
    
if __name__ == "__main__":

    ### Plotting from console ###
    import argparse
    parser = argparse.ArgumentParser(description="Plots the domain and learned model")
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
          