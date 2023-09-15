# Deep Reinforcement Learning for solving Partial Differential Equations

## Usage

Setup your PDE, domain, boundary and initial conditions in .py file
See parameters.py for the default setup

In a jupyter notebook, import the main file by using the line

`import DRLPDE.main`

and run the program by calling the name of the .py file (excluding the .py extension)

`model = DRLPDE.main.solvePDE('my_pde')`

Other arguments can be passed to modify the training parameters. 
The default parameters can be found in the define_solver_parameters function of DRLPDE.main.py 

main.ipynb is a Jupyter notebook implementing this exactly.

## Pre-built examples:

#### JCPexample1
Steady Stokes flow in a rotating disk
`model = DRLPDE.main.solvePDE('JCPexample1')`
- Has analytic solution

#### JCPexample2
Unsteady Stokes flow in a rotating disk
`model = DRLPDE.main.solvePDE('JCPexample2')`
- Has analytic solution

#### JCPexample3
Steady Stokes flow in rotating sphere
`model = DRLPDE.main.solvePDE('JCPexample3')`
- Has analytic solution

#### JCPexample4
Poiseuille Flow in a circular pipe
`model = DRLPDE.main.solvePDE('JCPexample4')`
- Has analytic solution

#### JCPexample5
Cavity Stokes Flow
`model = DRLPDE.main.solvePDE('JCPexample5')`
- No analytic solution

#### JCPexample6
Flow Past Disk
`model = DRLPDE.main.solvePDE('JCPexample6')`
- No analytic solution