
### TODO list

  - Statistics
    + Weighting of boundary and initial condition
      - Cavity flow: lambda = 1e2/dt, lambda_bdry = 1e0 works
    + Girsanov Theorem
    + Solid boundaries
  
  - Other problems
    + Incompressible Continuum Mechanics
    + Keller-Segel Model
    + Heat Transfer
    + Convection Diffusion
    + Full Navier-Stokes, pressure driven
    + BC: Neumann/Robin/Flux

- Different boundary types
  + curves (x, f(x)), (g(y), y)
  + cubic splines given a sequence of points (Bezier curves)
  + parametric curves ( f(t), g(t) )
  
- 3D space boundary types
  + surfaces (x, y, f(x,y)), (x, g(x,z), z), (h(y,z), y, z)
  + parametric surfaces ( f(s,t), g(s,t), h(s,t) )
  
- How to evaluate integral in reaction and forcing terms (right now using trapezoidal rule)
  
- Plotting
  + Unsteady videos and tracers in fluid