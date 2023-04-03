
### High Priority

+ Scientific Journals
  - Journal of Machine Learning
  - Journal of Computational Physics
  - SISC

+ List of experiments
  - Domain Decomposition Hybrid Method
    
  - Neural Network Parameters
    + Optimizer: LBGS vs ADAM
    + Architecture: Recurrent, Transformers, Reverse Diffusion
  
  - Statistics
    + Weighting of boundary and initial condition
      - Cavity flow: lambda = 1e2/dt, lambda_bdry = 1e0 works
    + Weighted number of points to achieve similar Mean Minimum Distance
    + Importance Sampling
    + Girsanov Theorem (Might affect variance?)
    + Walk or not
    + Solid boundaries
  
  - Other problems
    + Incompressible Continuum Mechanics
    + Belal's problem
    + Full Navier-Stokes, pressure driven
    + BC: Neumann/Robin/Flux


### Low Priority

- Different boundary types
  + curves (x, f(x)), (g(y), y)
  + cubic splines given a sequence of points (Bezier curves)
  + parametric curves ( f(t), g(t) )
  
- 3D space boundary types
  + spheres, cylinders
  + surfaces (x, y, f(x,y)), (x, g(x,z), z), (h(y,z), y, z)
  + parametric surfaces ( f(s,t), g(s,t), h(s,t) )
  
- How to evaluate integral in reaction and forcing terms (right now using trapezoidal rule)
  
- Plotting
  + Unsteady videos and tracers in fluid
  
- Split domain into 
  + For N boundaries, N+1 regions (close to i-th boundary)
  + Quick method of checking if walker is in which boundary
  + Tree structure for optimal region checking 