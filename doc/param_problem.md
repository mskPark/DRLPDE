# Guide to DRLPDE_param_problem.py file

## Save/Load model


## Physical Parameters

I recommend making including other physical parameters 
eg. lidspeed in Cavity Flow, inlet velocity in Flow Past Disk
Remember to make it a global variable

## Partial Differential Equation




## Boundary and Initial Conditions




## Making the Domain

First need a bounding box: $[x_1, x_2] \times [y_1, y_2] \times [z_1, z_2]$ that contains your domain

Use boundary classes to define the boundaries of your domain

#### 2D boundary classes

#### 3D boundary classes

#### Periodic boundaries

'variable' = 'x', 'y', 'z'
'base' is the bottom, if variable is below, then send to top
'top' is the top, if variable is above, then send to base


