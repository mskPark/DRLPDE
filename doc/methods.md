# Pseudocode for each different methods involved

## Evaluate the PDE

#### Reinforcement Learning
- Move walkers
    + with respect to SDE corresponding to PDE
- Calculate exits
    + Bisection method
    + TODO: bubble wrapping for bias reduction
- Evaluate at new location
    + Incorporate forcing term
    + Incorporate reaction term

#### Finite Difference
- Evaluate PDE
    + Approximate Derivatives with a stencil at each point
    + Be able to do different orders of approximation
    + For time-dependent problems, incorporate implicit and explicit methods
    + Distinguish between the data vs training point
- Boundary value
    + Spacing near boundary should line up the stencil to be on the bdry
    + Use the bdry value at the stencil point on the bdry

## Importance Sampling

#### Rejection Sampling


#### Metropolis-Hastings
