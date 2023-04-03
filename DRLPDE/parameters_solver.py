############## Parameters for the Solver ############

# Number of points
numpts = 2**12

# Number of batch points to send in
numbatch = 2**10

# Importance Sampling
resample_every = 300

# Random walkers make steps, exploring the domain
walk_every = 10

# Reweighting
reweight_every = 300

############## PDE Method ####################################

### Options: stochastic, autodiff
### TODO Finite Diff
method = "stochastic"

### Extra Parameters

# Martingale method

# Time step
dt = 1e-4

# exit tolerance
# CAREFUL: Because pytorch default float is float32, 1e-8 is machine epsilon
#          tol too low => max out recursion because of rounding errors 
tol = 1e-4

# Number of ghost
num_ghost = 128

# Hybrid method
# num_mesh = number of mesh points in each direction (equal for now)
# numpts = num_mesh - 2
num_mesh = 20
solvemesh_every = 50
h = 1/(num_mesh -1)


# FiniteDiff
# dt, tol, num_points, discretization

############## Deep Learning Parameters #######################

# Training steps
trainingsteps = 300
print_every = 100

# Neural Network Architecture
nn_depth = 20
nn_width = 4

# Parameters for ADAM optimizer
learningrate = 1e-2
adambeta = (0.9,0.999)
weightdecay = 0

