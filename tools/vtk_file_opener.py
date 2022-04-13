### vtk file open


import os
import sys
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

Nx = 256 # Number of points along x-axis - Does not count end point
Ny = 160 # Number of points along y-axis - Does not count end point
N_channel = 128 # 160*0.8
Lx = 2.5
Ly = 1.25
width = Ly*0.8
### Ly goes from [-1.25, 1.25] so 0.25 is outside pipe

Tfinal = 2.5
dt = 5.0e-5 
print_every = 500
mu = 1


reynolds_folder = 'Flow_Past_Disk_High_Reynolds'

#Just_open = False


num_files = int(Tfinal/dt/print_every)

u = np.zeros((Ny,Nx,num_files))
v = np.zeros((Ny,Nx,num_files))

### Modify vtk files to remove first 12 lines
for jj in range(0,num_files):
    index = '0' + str(jj)
    with open('C:/Users/Kevin/Documents/pytorch-stokes/vtk_files/' + reynolds_folder + '/u.00' + index[-2:] +'.vtk', 'r') as f:
        contents = f.readlines()
        contents = contents[12:]
        with open('C:/Users/Kevin/Documents/pytorch-stokes/vtk_files/' + reynolds_folder + '/vel.00' + index[-2:] +'.vtk', 'a') as g:
            for kk in range(len(contents)):
                g.write(contents[kk])

sys.exit()

for ii in range(0,num_files):
    index = '0' + str(ii)
    data = np.loadtxt('C:/Users/Kevin/Documents/pytorch-stokes/vtk_files/' + reynolds_folder + '/vel.00' + index[-2:] +'.vtk')

    data = data.reshape(-1).reshape((Nx*Ny,3))
    u[:,:,ii] = data[:,0].reshape(Ny,Nx)
    v[:,:,ii] = data[:,1].reshape(Ny,Nx)

u = u[16:N_channel + 16 + 1,:,:]
v = v[16:N_channel + 16 + 1,:,:]

u = np.concatenate( (u, u[:,0,None,:]), axis=1)
v = np.concatenate( (v, v[:,0,None,:]), axis=1)

u_x = (u[1:128,2:,:] - u[1:128,0:255,:])/256/2
u_y = (u[2:,1:256,:] - u[0:127,1:256,:])/128/2

div_u = u_x + u_y

sys.exit()

### TODO Plot

x1plot, x2plot = np.meshgrid(np.linspace(0,Lx,Nx+1), np.linspace(-width/2, width/2,N_channel + 1))
mag_vel = np.sqrt(u**2 + v**2)

circle_plot = np.linspace(0,2*math.pi, 30)
x_circle = width/6*np.cos(circle_plot) + width/2
y_circle = width/6*np.sin(circle_plot)

imagesfolder = "images/"
imagefilename = 'IBM_'  + reynolds_folder
framerate=5
video_name = imagesfolder + imagefilename + '.avi'
fourcc = cv2.VideoWriter_fourcc(*'MP4V') 

max_u_vel = np.amax(u)
min_u_vel = np.amin(u)
max_v_vel = np.amax(v)
min_v_vel = np.amin(v)


plt.close('all')
fig = plt.figure(figsize=(20,20))
fig.suptitle('Immersed Boundary Solution of Navier-Stokes with ' + r'\mu = {:f}'.format(mu) )

for ii in range(0,num_files):
    ax1 = fig.add_subplot(2,2,1,projection='3d')
    ax1.plot_surface(x1plot, x2plot, u[:,:,ii], cmap=plt.cm.coolwarm)
    ax1.set_title("IBD Approximation of u1 velocity")
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlim(min_u_vel, max_u_vel)
    ax1.view_init(elev=20, azim=-38)
    
    ax2 = fig.add_subplot(2,2,2,projection='3d')
    ax2.plot_surface(x1plot, x2plot, v[:,:,ii], cmap=plt.cm.coolwarm)
    ax2.set_title("IBD Approximation of u2 velocity")
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlim(min_v_vel, max_v_vel)
    ax2.view_init(elev=20, azim=-38)
    
    ax3 = fig.add_subplot(2,2,(3,4))
    #ax3.quiver( x1plot, x2plot, u[:,:,ii], v[:,:,ii])
    ax3.set_title("Contour of Magnitude of Velocity, Time = " + '{:.4f}'.format(ii*print_every*dt))
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.plot(x_circle, y_circle, ls='-')
    contour = ax3.contourf(x1plot, x2plot, mag_vel[:,:,ii], alpha=0.5)
    cbar = plt.colorbar(contour)
    
    
    filename = imagesfolder + imagefilename + str(ii) + ".png"
    if os.path.isfile(filename):
        os.remove(filename)
    fig.savefig(filename)
    
    if ii==0:
        frame = cv2.imread(imagesfolder + imagefilename + str(ii) + ".png") 
        # setting the frame width, height width 
        # the width, height of first image 
        height, width, layers = frame.shape   
        video = cv2.VideoWriter(video_name, fourcc, framerate, (width, height)) 
        video.write(cv2.imread(imagesfolder + imagefilename + str(ii) + ".png")) 
    else:
        video.write(cv2.imread(imagesfolder + imagefilename + str(ii) + ".png"))

    fig.clf()
    # Deallocating memories taken for window creation 
     
video.release()  # releasing the video generated     
cv2.destroyAllWindows()
#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(1,1,1)
#ax.scatter(y[0:512,0], y[0:512,1])
#plt.show()