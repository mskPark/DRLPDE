# Deep Reinforcement Learning for solving Partial Differential Equations

This project learns a neural network that solves elliptic/parabolic PDEs with initial and boundary conditions in a bounded domain.
The reinforcement learning objective is the martingale condition of the stochastic process that corresponds to the PDE

## Usage

Setup your PDE, domain, boundary and initial conditions in the DRLPDE_param_problem.py file
    (Optional) Setup your deep learning parameters in the DRLPDE_param_solver.py file

Run the DRLPDE_main file from the command line: python DRLPDE_main.py

If you prefer a jupyter notebook environment, use DRLPDE_main.ipynb 

## Pre-built examples:

You can run any of the pre-built examples by calling its name: python DRLPDE_main.py example 1
    
#### Example1

#### Example2

#### Example3