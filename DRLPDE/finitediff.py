### Numerical Derivatives
###   At random points, perform finite difference to approximate derivatives
###   Combine Linear Multistep methods (Adam Bashforth) and Runge-Kutta methods for time dependent problems
###   Can utilize implicit formulas for better stability?

def forward_diff(x, model, h):
    ###
    ### 1/h * ( model(x + h) - model(x) )

    pass

def backward_diff(y,x, h):
    ###
    ### 1/h * ( model( x - h) - model(x) )
    pass

def center_diff(y,x, h):
    ###
    ### 1/2/h * ( model(x + h) - model(x - h) )
    pass
