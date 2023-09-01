## Deep Reinforcement Learning of solutions to PDEs

This project provides a numerical method to solve partial differential equations with boundary and/or initial conditions.

Explore the domain, evaluate how well we're doing, improve the estimate

#### Features
- Mesh-free
- Hyperparameters to train a family of solutions

## Pseudocode

Define the problem
- PDE
  + Linear Elliptic/Parabolic
    Drift, Reaction, Forcing, Viscosity
  + viscous Burgers
    Forcing, Viscosity
  + incompressible Navier Stokes
    Forcing, Viscosity

- Learning Method
  + Martingale
  + AutoDiff

- Domain
  + Dirichlet walls
    BC
  + Periodic ends
  + Solid walls
    BC
  + Inlets + Outlets
    BC


Initialization
- Domain Class/Function:
  + Input: Bounding Box, List of boundaries, List of boundary conditions
  + Output: Boundary Classes

- Data Generation Class (with DataLoader):
  + Input: Boundary classes, model
  + Output: Batches of points

  + Functions: Make Target, Resample 

- Boundary/Initial Condition Class (with DataLoader):
  + Input: Boundary classes, model
  + Output: Batches of points + Boundary values

  + Functions: Make Target, Resample (?)

Training Step
- In batches:
  + Make Loss at each point (interior and boundary)
    - Send out random walk, calculate exits, evaluate 
  + Calculate Max
  + Do resampling step if necessary
    Consider resampling domain points and boundary points
  + Average the loss, calculate backward
- Optimization Step

Pre-training Visualization:
- 2D Image of the domain: 
  + Plot boundaries
  + Fill in domain - for meshgrid of boundingbox, color binary
  + Display whether unsteady or not
- 3D Not sure

Post-training Visualization:
- 2D contour plots: Use tricontour
- For flow problems: movie with RK4 tracers


## Organization
+ docs
- md files

main file
param_PDE file
param_solver file

DefineDomain module
CalculateLoss module
NeuralNetworks module

- (DRLPDE steady)
- (DRLPDE unsteady)
- (DRLPDE continuation)

+ domain package
  - init 
  - Domain2D module
  - Domain3D module

+ method package
  - init
  - FiniteDifference module
  - Stochastic module
  - AutomaticDifference module

+ PostProcessing package
  - init
  - 

+ test
  - JCPexample1
  - JCPexample1
  - JCPexample1
  - JCPexample1
  - JCPexample5
  - JCPexample6

savedmodels

plots

tools