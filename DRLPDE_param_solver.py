############## Walker and Boundary Parameters ############

# Time step
dt = 1e-4

# exit tolerance
tol = 1e-6

# Number of walkers
num_walkers = 2**14
num_ghost = 128
num_batch = 2**12

# Update walkers
# Options: 
#    move -- moves walkers to one of their new locations
#    remake -- remake walkers at each training step
#    fixed -- keeps walkers fixed
update_walkers = 'remake'
update_walkers_every = 100

# Number of boundary points 
num_bdry = 2**12
num_batch_bdry = 2**9

# Number of initial points
num_init = 2**12
num_batch_init = 2**9

############## Training Parameters #######################

# Training epochs
num_epoch = 2000
update_print_every = 1000

# Neural Network Architecture
nn_depth = 60
nn_width = 4

# Weighting of losses
lambda_bell = 1e-1 # 1e-2/dt
lambda_bdry = 1e2
lambda_init = 1e-1

# Learning rate
learning_rate = 1e-2
adam_beta = (0.9,0.999)
weight_decay = 0

