############## Parameters for the Solver ############

# Number of points
numpts = 2**12

# Number of batch points to send in
numbatch = 2**10

############## PDE Method ####################################

# Options: stochastic, autodiff
# TODO Finite Diff, Direct
method = "stochastic"

### Stochastic method parameters

# Time step
dt = 1e-4

# Number of ghost
num_ghost = 128

# Random walkers make steps, exploring the domain
walk_every = 1000

# exit tolerance
# CAREFUL: Because pytorch default float is float32, 1e-8 is machine epsilon
#          tol too low => max out recursion because of rounding errors 
tol = 1e-4

### Finite Difference method parameters
# dt, tol, num_points, discretization

############## Deep Learning Parameters #######################

# Training steps
trainingsteps = 300
print_every = 100

# Neural Network Architecture
nn_depth = 60
nn_width = 4

# Parameters for ADAM optimizer
learningrate = 1e-2
adambeta = (0.9,0.999)
weightdecay = 0

# Try LBFGS
# torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)


############### Adaptive Parameters ###########################

### To disable, give a large number
# Importance Sampling
resample_every = 200

# Reweighting
reweight_every = 200

### Hybrid method parameters
# Disable in parameters by not including mesh
# num_mesh = number of mesh points in each direction (equal for now)
# numpts = num_mesh - 2
num_mesh = 20
solvemesh_every = 50
h = 1/(num_mesh -1)
