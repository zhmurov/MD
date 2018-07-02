import numpy as np
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# TODO ========
frames = np.arange(9,601,1)
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
    plt.xlabel('Time, ns',fontsize=20)

    stdev = []
    for i in frames:
        meandev = np.loadtxt('output1/meandev_disp%d.dat'%i)
        stdev.append((meandev[p, 1])**(1.0/2.0))

    plt.plot(frames/10.0, stdev, color='black', linewidth=1.4)
    plt.plot([0,60], [stdev[-1],stdev[-1]], color='red', linestyle='--')

    plt.legend()
    plt.savefig("graphs1/pair_%d"%p, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()

