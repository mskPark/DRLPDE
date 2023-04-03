# Deep Reinforcement Learning for solving Partial Differential Equations

## Features
- Types of PDEs:
    + Elliptic
    + Parabolic
    + Stokes flow
    + Navier-Stokes
- Types of Domains:
    + 2D -- An intersection of 2D shapes (half-planes, disks)
    + 3D -- An intersection of 3D shapes (half-spaces, spheres, cylinders)
- Boundary Conditions:
    + Dirichlet
    + Pressure difference
- Method:
    + Automatic differentiation
    + Martingale condition

## Usage

Setup your PDE, domain, boundary and initial conditions in the DRLPDE_param_problem.py file
(Optional) Setup your deep learning parameters in the DRLPDE_param_solver.py file

Run the DRLPDE.main file from the command line: python DRLPDE.main.py
Or use the jupyter notebook 

## Pre-built examples:

#### JCPexample1
Steady Stokes flow in a rotating disk
- Has analytic solution

#### JCPexample2
Unsteady Stokes flow in a rotating disk
- Has analytic solution

#### JCPexample3
Steady Stokes flow in rotating sphere
- Has analytic solution

#### JCPexample4
Poiseuille Flow in a circular pipe
- Has analytic solution

#### JCPexample5
Cavity Stokes Flow
- No analytic solution

#### JCPexample6
Flow Past Disk
- No analytic solution