- Split domain into 
  + For N boundaries, N+1 regions (close to i-th boundary)
  + Quick method of checking if walker is in which boundary
  + Tree structure for optimal region checking

- Importance sampling
  + Use loss to remove points -- low loss, can remove (with probability?)
  + Maybe remove a fraction and resample to keep same number
  
- Different boundary types
  + curves (x, f(x)), (g(y), y)
  + cubic splines given a sequence of points (Bezier curves)
  
- 3D space boundary types
  + spheres, cylinders
  + surfaces (x, y, f(x,y)), (x, g(x,z), z), (h(y,z), y, z)
  
- Girsanov Theorem

- Finding an actual point on the boundary instead of closest up to a tolerance

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
   