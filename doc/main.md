## Reinforcement Learning of solutions to PDEs in irregular domains

#### Explore the domain, evaluate how well we're doing, improve the estimate

Goals
- Mesh-free
- Easy to use
- Include hyperparameters to train for a class of PDEs

Features
- Improve stability
- Importance sampling
- Guaranteed accuracy (Very hard, dependent on a lot of parameters)
    + Size/Architecture of neural netowk

Domain description
- Boundary curves/surfaces (inside/outside by signed distance function) 
  TODO: Use parametric equations vs level set
  Parametric equations
  + Sample and move along boundary easily
  - Hard to make signed distance function (specify normal vector, have to do an fsolve)
  Level set
  + Easy to make signed distance function
  - Hard to move along boundary (sampling is doable for type 1 / type 2 curves)



## Ideas

Keep steady vs unsteady separate
  + Generating point data is easier this way. Keep them separate functions

Continuation methods
  + Solve the linear problem. Use that as an initial guess for the non-linear problem.
  + Need a ramp up parameter

## Pseudo-code

Main Training Function
- param_problem
- param_solver
- use_cuda

Initialization Step
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
docs
- md files

src
- DRLPDE main file
- DefineDomain file
- CalculateLoss file

- (DRLPDE steady)
- (DRLPDE unsteady)
- (DRLPDE continuation)

+ DomainTypes folder
+ MethodPDE folder
  - FiniteDifference
  - RandomWalk
  - AutomaticDifference
  - ContinuationMethod
+ NeuralNetworks

test
- JCPexample1
- JCPexample1
- JCPexample1
- JCPexample1
- JCPexample5
- JCPexample6

savedmodels

plots

tools
