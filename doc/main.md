# Pseudo-code of the main solver file

Initialization Step
- Setup boundaries
- Setup neural network
    + For incompressible flow, use vector potential
- Setup data:
    + Interior Random Walkers
    + Boundary points
    (Use DataLoader framework)

Training Step
- Move walkers
    + with respect to SDE corresponding to PDE
- Calculate exits
    + Bisection method
    + No bubble wrapping for bias reduction
- Evaluate at new location
    + Incorporate forcing term
    + Incorporate reaction term

##