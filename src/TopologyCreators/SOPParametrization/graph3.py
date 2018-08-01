import numpy as np
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# TODO ========
iteration = np.arange(1,256,1)
Npairs = 32
# =============


pair = range(Npairs)

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
matplotlib.rcParams.update(params)

for p in pair:
    plt.ylabel('Standard deviation, angstr',fontsize=20)
    plt.xlabel('Iteration',fontsize=20)

    straight_line = []
  
    dev0 = np.loadtxt('output3/meandev_disp0.dat')
    dev0 = (dev0[p, 1])**(1.0/2.0)
    
    dev = []
    for i in iteration:
        std_dev = np.loadtxt('output3/meandev_disp%d.dat'%i)
        std_dev = (std_dev[p, 1])**(1.0/2.0)
        dev.append(std_dev)
    
    plt.plot(iteration, dev, color='black', label=str('Coarse-grained'), linewidth=1.4)
    
    #lastdev = np.loadtxt('output3/meandev_disp512.dat')
    #lastdev = (lastdev[p, 1])**(1.0/2.0)
    plt.plot([0,256], [dev0,dev0], color='red', label=str('Full-atomic'), linestyle='--')

    plt.legend()
    plt.savefig("graphs3/pair_%d"%p, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
