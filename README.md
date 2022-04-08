# Deep Reinforcement Learning for solving Partial Differential Equations

An unsupervised deep reinforcement learning framework that solves Initial Boundary Value Problems (IBVP) of Partial Differential Equations (PDE)

A neural network is trained to satisfy the martingale condition corresponding to the PDE

## Features:
- Types of PDEs:
    + Elliptic
    + Parabolic
    + Stokes flow
    + Navier-Stokes
    + (TODO) Non-linear through Newton's method
- Types of Domains:
    + 2D -- An intersection of 2D shapes (half-planes, disks)
    + 3D -- An intersection of 3D shapes (half-spaces, spheres, cylinders)

## Usage

Setup your PDE, domain, boundary and initial conditions in the DRLPDE_param_problem.py file
(Optional) Setup your deep learning parameters in the DRLPDE_param_solver.py file

Run the DRLPDE_main file from the command line: python DRLPDE_main.py

## Pre-built examples:

You can run any of the pre-built examples by calling its name in python: DRLPDE_main.py -example name_of_example

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