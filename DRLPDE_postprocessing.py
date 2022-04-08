###
### Plotting file 
###

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300

import math
import torch

num_plot_xdim = 128
num_plot_ydim = 128
num_plot_bdry = 64
num_plot_rand = 256

import DRLPDE_nn
import DRLPDE_functions

###
### PostProcessing function
###    Plot domain
###    Steady States:
###       Model + Analytic solution contour plot for each output
###    Unsteady States:
###       Model + Analytic solution contour plot for given time points
###       TODO: video
###    Error values:
###       L^2 and L^{\infty} error given


def postprocessing(param='DRLPDE_param_problem',
                   use_model=''):
    
    plot_title = True
    
    print("Plotting now")
    
    ###
    ### Unpack variables
    ###
    
    import importlib
    if param=='DRLPDE_param_problem':
        DRLPDE_param = importlib.import_module("DRLPDE_param_problem")
        print("Loading parameters from default location: DRLPDE_param_problem.py")
    else:
        DRLPDE_param = importlib.import_module("." + param, package='examples')
        print("Loading parameters from " + param + '.py')
        
    boundingbox = DRLPDE_param.boundingbox
    my_bdry = DRLPDE_param.my_bdry
    output_dim = DRLPDE_param.output_dim
    exists_analytic_sol = DRLPDE_param.exists_analytic_sol
    is_unsteady = DRLPDE_param.is_unsteady
    
    # If unsteady, include time range in bounding box
    if is_unsteady:
        boundingbox.append(DRLPDE_param.time_range)
        plot_times = DRLPDE_param.plot_times
    
    if exists_analytic_sol:
        plot_levels = DRLPDE_param.plot_levels
        plot_ticks = [DRLPDE_param.plot_levels[0], 0, DRLPDE_param.plot_levels[-1]]
    else:
        plot_levels = 100
        plot_ticks = 3
    
    boundaries = DRLPDE_functions.make_boundaries(my_bdry)
    
    ###
    ### Plot the domain
    ###
    fig_domain, ax_domain = plt.subplots()

    Xbdry = []
    
    for bdry in boundaries:
        X_each_bdry = bdry.plot_bdry(num_plot_bdry)
        Xbdry.append( X_each_bdry )
        
        ax_domain.plot( X_each_bdry[:,0].numpy(), X_each_bdry[:,1].numpy(), 'k-' )
        
        X_rep_bdry = bdry.plot_point()
        
        ax_domain.scatter(X_rep_bdry[0].numpy(), X_rep_bdry[1].numpy(), color='green', marker='o' )
    
    
    ax_domain.set_title('Domain')
    
    fig_domain.savefig("plots/" + param + "_domain.png")
    print("Domain plot saved in " + "plots/" + param + "_domain.png")
    
    ###
    ### Setup the boundary points for solution plots
    ###
    
    Xbdrypoints = torch.cat( Xbdry, dim=0)
    
    if is_unsteady:
        Xbdrypoints = torch.cat( ( Xbdrypoints, torch.zeros( (Xbdrypoints.size(0),1) ) ), dim=1)
        
    Xplot = torch.cat( (Xbdrypoints, 
                        DRLPDE_functions.generate_interior_points(num_plot_rand, boundingbox, boundaries)), dim=0 )
    
    ###
    ### Plotting
    ###
    if is_unsteady:
        
        if use_model:
            plot_unsteady_model(param, use_model, output_dim, Xbdry, Xplot, plot_levels, plot_ticks, plot_title, plot_times)
            
            print("Model solution plot saved in " + "plots/" + param + "_model.png")

        if exists_analytic_sol:
            plot_unsteady_analytic(param, DRLPDE_param.true_solution, output_dim, 
                                   Xbdry, Xplot, plot_levels, plot_ticks, plot_title, plot_times)
            
            print("Analytic solution plot saved in " + "plots/" + param + "_analytic.png")
    else:
        
        if use_model:
            plot_steady_model(param, use_model, output_dim, Xbdry, Xplot, plot_levels, plot_ticks, plot_title)
            
            print("Model solution plot saved in " + "plots/" + param + "_model.png")
            
        if exists_analytic_sol:
            plot_steady_analytic(param, DRLPDE_param.true_solution, output_dim, 
                                 Xbdry, Xplot, plot_levels, plot_ticks, plot_title)
            
            print("Analytic solution plot saved in " + "plots/" + param + "_analytic.png")

    ### TODO: plot difference between model and true solution        
    
    ###
    ### Error
    ###
    ### TODO: Make error function for imported data solutions
    if use_model and exists_analytic_sol:
        
        model = torch.load("savedmodels/" + use_model + ".pt").to('cpu')
        true_sol = DRLPDE_param.true_solution
        L2error, Linferror = error(boundingbox, boundaries, model, true_sol)

        print('L2 error is {:.3e}\nLinf error is {:.3e}'.format(L2error, Linferror ))

        
        
        
######
###### Plotting for Steady state problems
######

###
### Contour plot for model solution
###

def plot_steady_model(param, use_model, output_dim, Xbdry, Xplot, plot_levels, plot_ticks, plot_title):
        
    model = torch.load("savedmodels/" + use_model + ".pt").to('cpu')
    
    Usol = []
    
    ### Plot boundaries first
    fig_model, ax_model = plt.subplots(nrows=output_dim, ncols=1, figsize=[6.4, output_dim*5.0])

    for ii in range(output_dim):
        for jj in range(len(Xbdry)):
            ax_model[ii].plot( Xbdry[jj][:,0].numpy(), Xbdry[jj][:,1].numpy(), 'k-' )

    ### Plot solution
    Uplot = torch.split( model(Xplot.requires_grad_(True)), 1, dim=1 )

    for ii in range(output_dim):
        surf = ax_model[ii].tricontourf(Xplot[:,0].detach().numpy(), 
                                        Xplot[:,1].detach().numpy(), 
                                        Uplot[ii][:,0].detach().numpy(), 
                                        cmap=plt.cm.viridis, levels=plot_levels)
        plt.colorbar(surf, ticks=plot_ticks, ax=ax_model[ii])

    if plot_title:
        for ii in range(output_dim):
            ax_model[ii].set_title('Plot of output component ' + str(ii))

    fig_model.savefig("plots/" + param + "_model.png")
    
    
    Usol.append(Uplot)
    
    return Usol

    
###
### Contour plot for analytic solution
###    
    
def plot_steady_analytic(param, true_sol, output_dim, Xbdry, Xplot, plot_levels, plot_ticks, plot_title):
    
    Usol = []
    
    fig_analytic, ax_analytic = plt.subplots(nrows=output_dim, ncols=1, figsize=[6.4, output_dim*5.0])
    
    ### Plot boundary points
    for ii in range(output_dim):
        for jj in range(len(Xbdry)):
            ax_analytic[ii].plot( Xbdry[jj][:,0].numpy(), Xbdry[jj][:,1].numpy(), 'k-' )

    ### Plot the solution
    Utrueplot = torch.split( true_sol(Xplot), 1, dim=1 )

    for ii in range(output_dim):
        surf = ax_analytic[ii].tricontourf(Xplot[:,0].detach().numpy(), 
                                           Xplot[:,1].detach().numpy(), 
                                           Utrueplot[ii][:,0].detach().numpy(), 
                                           cmap=plt.cm.viridis, levels=plot_levels)
        plt.colorbar(surf, ticks=plot_ticks , ax=ax_analytic[ii])

    if plot_title:
        for ii in range(output_dim):
            ax_analytic[ii].set_title('Plot of analytic solution component ' + str(ii))
    
    fig_analytic.savefig("plots/" + param + "_analytic.png")
    
    Usol.append(Uplot)
    
    return Usol

######
###### Plotting for Unsteady state problems
######

###
### Contour plot for model solution
###

def plot_unsteady_model(param, use_model, output_dim, Xbdry, Xplot, plot_levels, plot_ticks, plot_title, plot_times):
        
    model = torch.load("savedmodels/" + use_model + ".pt").to('cpu')
    
    Usol = []
    
    for jj in range(len(plot_times)):

        fig_model, ax_model = plt.subplots(nrows=output_dim, ncols=1, figsize=[6.4, output_dim*5.0])
        
        ### Plot boundaries first
        for ii in range(output_dim):
            for jj in range(len(Xbdry)):
                ax_model[ii].plot( Xbdry[jj][:,0].numpy(), Xbdry[jj][:,1].numpy(), 'k-' )
                
        ### Plot solution
        X.requires_grad = False
        Xplot[:,-1] = plot_times[jj]
        X.requires_grad = True
        Uplot = torch.split( model(Xplot), 1, dim=1 )

        for ii in range(output_dim):
            surf = ax_model[ii].tricontourf(Xplot[:,0].detach().numpy(), 
                                            Xplot[:,1].detach().numpy(), 
                                            Uplot[ii][:,0].detach().numpy(), 
                                            cmap=plt.cm.viridis, levels=plot_levels)
            plt.colorbar(surf, ticks=plot_ticks, ax=ax_model[ii])

        if plot_title:
            for ii in range(output_dim):
                ax_model[ii].set_title('Plot of output component ' + str(ii) + 'at time ' + str(plot_times[jj]) )

        fig_model.savefig("plots/" + param + "_model" + "_time_" + str(plot_times[jj]) +".png")
        
        Usol.append(Uplot)
        
    return Usol
    
###
### Surface plot for analytic solution
###    
    
def plot_unsteady_analytic(param, true_sol, output_dim, Xbdry, Xplot, plot_levels, plot_ticks, plot_title, plot_times):
    
    Usol = []
    
    for jj in range(len(plot_times)):
    
        fig_analytic, ax_analytic = plt.subplots(nrows=output_dim, ncols=1, figsize=[6.4, output_dim*5.0])

        ### Plot boundary points
        for ii in range(output_dim):
            for jj in range(len(Xbdry)):
                ax_analytic[ii].plot( Xbdry[jj][:,0].numpy(), Xbdry[jj][:,1].numpy(), 'k-' )

        ### Plot the solution
        X.requires_grad = False
        Xplot[:,-1] = plot_times[jj]
        Utrueplot = torch.split( true_sol(Xplot), 1, dim=1 )

        for ii in range(output_dim):
            surf = ax_analytic[ii].tricontourf(Xplot[:,0].detach().numpy(), 
                                               Xplot[:,1].detach().numpy(), 
                                               Utrueplot[ii][:,0].detach().numpy(), 
                                               cmap=plt.cm.viridis, levels=plot_levels)
            plt.colorbar(surf, ticks=plot_ticks , ax=ax_analytic[ii])

        if plot_title:
            for ii in range(output_dim):
                ax_analytic[ii].set_title('Plot of analytic solution component ' + str(ii))

        fig_analytic.savefig("plots/" + param + "_analytic"+ "_time_" + str(plot_times[jj]) + ".png")
        
    Usol.append(Uplot)
        
    return Usol

######
###### Calculate error
######

###
### L^2 and L^{\infty} error given a (true solution) function
###
### TODO: May make a separate function in functions file 
###    Give error between previous model -> To use during training, see the cauchy sequence
###
### Note: If true solution given as datapoints, then make a separate jupyter notebook

def error(boundingbox, boundaries, model, true_sol, time_range=False):          
        
    ### Error plots using Monte Carlo Integration
    
    # Tolerance for error
    error_tol = 1e-2
    
    # Confidence: 1 = 68%, 2 = 95%, 3 = 99.75%, 5 = 99.99995%
    #    Number of points increases by (confidence/error_tol)^2
    confidence = 2
    
    num_error = int( (confidence/error_tol)**2 )

    Xerror = torch.empty( (num_error, len(boundingbox)) )
    
    vol_box = 1

    for kk in range(len(boundingbox)):
        Xerror[:,kk] = (boundingbox[kk][1] - boundingbox[kk][0])*torch.rand( (num_error)) + boundingbox[kk][0]
        vol_box = vol_box*(boundingbox[kk][1] - boundingbox[kk][0])
        
    outside = torch.zeros( Xerror.size(0), dtype=torch.bool)

    for bdry in boundaries:
        outside += bdry.dist_to_bdry(Xerror) > 0

    ### Volume of Region = Fraction of points that got killed off * Volume of Domain
    vol_domain = ( 1 - torch.sum(outside)/num_error )*vol_box
    
    Xerror[outside, :] = DRLPDE_functions.generate_interior_points(torch.sum(outside), boundingbox, boundaries).requires_grad_(True)
    
    True_minus_trained = (true_sol(Xerror) - model(Xerror)).detach()

    L2error = ( vol_domain*torch.sqrt( ( torch.sum(True_minus_trained**2) )/num_error ) ).numpy()
    Linferror = torch.max( torch.abs( True_minus_trained ) ).numpy()

    return L2error, Linferror

    
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
        
    postprocessing(param, model)
          