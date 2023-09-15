### For copy paste purposes

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[6.4, 5.0])

plt.tight_layout(pad=0.75)

contour = ax[0,0].contourf( x1, x2, u, levels, cmap=plt.cm.viridis)

plt.cm.coolwarm

ax[0,0].set_title(r'$\tilde{u}$')

ax[0,0].tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False
)

colorbar0_param = fig.add_axes(
    [ax[0,0].get_position().x1 + 0.01,
     ax[0,0].get_position().y0,
     0.01,
     ax[0,0].get_position().height])
colorbar0 = plt.colorbar(contour0, ticks=level_tick, cax = colorbar0_param)
colorbar0.ax.tick_params(labelsize=10)

ax[0,0].text(-1.675,
             0,
             r'$t=0$',
             fontsize=10)

## Example of using torch viz to show the computational graph
from torchviz import make_dot
import DRLPDE.neuralnets
import torch

model = DRLPDE.neuralnets.ResNetNN([2,0,0], 1, 3, 20)

x = torch.randn((10,2))

make_dot(model(x), params = dict(model.named_parameters()))