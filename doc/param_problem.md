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


# Input (Provided by the user)
    #   boundingbox   - list of arrays
    #   list_of_walls - list of dictionaries
    #   periodic_ends - list of dictionaries
    #   inlet_outlet  - list of dictionaries
    #   solid_walls   - list of dictionaries
    #   initial_con   - function

    # Attributes:
    #   boundingbox - list of arrays
    #   wall        - list of wall classes
    #   periodic    - list of periodic ends classes
    #   inletoutlet - list of wall classes
    #   solid       - list of solid wall classes
    #   box         - list of finite difference operations
    #   volume      - number

    # Wall attributes:
    #   bc  - function that calculates boundary condition
    #   dim - dimension of boundary
    #   measure - length/area of boundary
    #   distance - function that calculates signed distance to wall

    # Periodic end attributes
    #   index: 0, 1, 2 for x, y, z respectively
    #   bot:   number
    #   top:   number

    # Inlet Outlet attributes
    #   bc: function that calculates pressure
    #   dim: dimension of inlet
    #   measure: length/area of boundary
    #   distance: function that calculates signed distance to inlet/outlet

    # Solid attirbutes:
    #   bc - the value of the function inside the solid
    #   dim - dimension of boundary
    #   measure - area/volume of solid
    #   distance - function that calculates signed distance to solid