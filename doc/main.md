# Pseudo-code of the main solver file

Arguments: 
- param_problem
- param_solver
- use_cuda

Initialization Step
- Parse param_problem file
- Parse param_solver file
- Import stuff
- Setup boundaries
- Setup neural network
    + For incompressible flow, use vector potential
- Setup points in domain (Use DataLoader framework)
    + Interior: Random points
    + Boundary: Random points
    (Use DataLoader framework)

- Setup training method
    + Calculate PDE
    + Calculate Loss at each point
    + Calculate Total loss
    + Optimization step

Training Step
- Evaluate training method once
- Importance Sampling method
  + Move walkers (Multiple options)
  + Calculate exits if necessary
- Train!