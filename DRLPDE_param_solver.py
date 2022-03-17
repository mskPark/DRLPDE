############## Walker and Boundary Parameters ############

# Time step
dt = 1e-4

# exit tolerance
tol = 1e-6

# Number of walkers
num_walkers = 2**12
num_ghost = 128
num_batch = 2**6

# Update walkers
# Options: 
#    move -- moves walkers to one of their new locations
#    remake -- remake walkers at each training step
#    fixed -- keeps walkers fixed
update_walkers = 'remake'
update_walkers_every = 1

# Number of boundary points 
num_bdry = 2**8
num_batch_bdry = 2**6

# Number of initial points
num_init = 2**8
num_batch_init = 2**6

############## Training Parameters #######################

# Training epochs
num_epoch = 3
update_print_every = 1

# Neural Network Architecture
nn_depth = 64
nn_width = 4

# Weighting of losses
lambda_bell = 1e-2/dt
lambda_bdry = 1e2
lambda_init = 1e2

# Learning rate
learning_rate = 1e-2
adam_beta = (0.9,0.999)
weight_decay = 1e-4

