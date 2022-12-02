
- Density of bdry points - area for disk affects how many points needed
  Currently the number of bdry points is picked arbitrarily. Automate the process
  Maybe have an accuracy requirement -> number of points needed?

- Train one step:
  + Check for errors
  + Do importance sampling
  + Compare interior error and boundary error
    - From the cavity flow example, the loss terms have to be of the same order of magnitude
    - If lambda_bdry is 1e2, then it fails to work.
    - But if lambda_bdry is 1e0, then it works well

- Split domain into 
  + For N boundaries, N+1 regions (close to i-th boundary)
  + Quick method of checking if walker is in which boundary
  + Tree structure for optimal region checking

- Number of boundary points
  + Utilize the length/area of the boundary to divide up the total number of points evenly
  + 

- Different boundary types
  + curves (x, f(x)), (g(y), y)
  + cubic splines given a sequence of points (Bezier curves)
  
- 3D space boundary types
  + spheres, cylinders
  + surfaces (x, y, f(x,y)), (x, g(x,z), z), (h(y,z), y, z)
  
- Girsanov Theorem

- How to evaluate integral in reaction and forcing terms (right now using right-end evaluation)

- Different boundary conditions
  + Neumann
  + Robin
  + Flux integral for outlet (and inlet?) flow
  + Periodic boundaries
  
- Plotting
  + Unsteady videos and tracers in fluid
  
- More neural network architectures
  + Recurrent networks
   