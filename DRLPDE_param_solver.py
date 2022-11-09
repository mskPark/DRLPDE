############## Parameters for the Solver ############

# Time step
dt = 1e-4

# exit tolerance
tol = 1e-8

# Solver Type
#    Martingale
#        dt, tol, num_walkers, num_ghost, update_walkers
#
#    AutoDiff
#        num_points
#
#    FiniteDiff
#        dt, tol, num_points, discretization

# Universal parameters
#     num_batch, update_points_every

# Number of walkers
num_walkers = 2**12
num_ghost = 128
num_batch = 2**12

# Update walkers
#    move -- moves walkers to one of their new locations
#    remake -- remake walkers at each training step
#    fixed -- keeps walkers fixed
update_walkers = 'remake'
update_walkers_every = 1

# TODO: Bdry and Initial points can be chosen automatically from number of interior points

# Number of boundary points 
num_bdry = 2**12
num_batch_bdry = 2**12

# Number of initial points
num_init = 2**12
num_batch_init = 2**12

############## Deep Learning Parameters #######################

# Training epochs
num_epoch = 1000
update_print_every = 100

# Neural Network Architecture
nn_depth = 60
nn_width = 4

# Weighting of losses
lambda_bell = 1e-2 # 1e-2/dt
lambda_bdry = 1e2
lambda_init = 1e-1

# Learning rate
learning_rate = 1e-2
adam_beta = (0.9,0.999)
weight_decay = 0

