# Guide to DRLPDE_param_problem.py file

## Save/Load model

## Specify the PDE
Linear with variable coefficient

Quasilinear (Navier-Stokes)

## Specify Boundary and Initial Condition functions

function with torch operations to evaluate points

## Parameters

#### Ordering: space, time, other

space dimension

unsteady vs steady flow

other parameters
    eg. lidspeed in Cavity Flow, inlet velocity in Flow Past Disk, viscosity

## Making the Domain

Bounding box: $[x_1, x_2] \times [y_1, y_2] \times [z_1, z_2]$ for space parameters

TimeRange: $[t0, t1]$ for time range

ParamRange: $[a0, a1]$ for parameter range

## Specify Boundaries

#### 2D boundary classes

#### 3D boundary classes

#### Periodic boundaries

'variable' = 'x', 'y', 'z'
'base' should match BoundingBox
'top' should match BoundingBox


