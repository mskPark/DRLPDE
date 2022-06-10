# Time step
dt = 5e-2
# exit tolerance
tol = 1e-8

# Number of walkers

numpts = 2**13

num_ghost = 512
num_batch = 2**9

num_fixed = 256

# 
use_true_vel = True
do_stopping_time = False

# Update walkers
# Options: 
#    move -- moves walkers to one of their new locations
#    remake -- remake walkers at each training step
#    fixed -- keeps walkers fixed
update_walkers = 'move'
update_walkers_every = 1

# Calculate true errors

calc_error_every = 100

############## Training Parameters #######################

# Training epochs
num_epoch = 3000

update_print_every = 1000

# Neural Network Architecture
nn_depth = 60
nn_width = 4

# Weighting of losses
lambda_bell = 1e0
lambda_fix = 1e-1

# Learning rate
learning_rate = 1e-3
adam_beta = (0.9,0.999)
weight_decay = 0


#L2 error = 0.0692
#Linf error = 0.0796
#L2 Relative error = 0.0590
#L2 norm true = 1.1718
#Error at origin = 0.0174

#### Other parameters

# time range = [0, 0.25]
# sampling = sqrt(rand) in time
# grad pressure term: Trapezoidal rule