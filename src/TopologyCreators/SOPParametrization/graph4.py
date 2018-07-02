import numpy as np
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# TODO ========
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

# TODO Axis X bounds
v_min = 2
v_max = 10


x = np.linspace(v_min, v_max, 512)

for p in pair:
    plt.ylabel('Probability density function',fontsize=20)
    plt.xlabel('Contact distance, angstr',fontsize=20)

    meandev0 = np.loadtxt('output4/meandev_disp0.dat')
    #normal distrivbution from mean(loc) and standard deviation(scale)
    n_pdf = norm.pdf(x, loc=meandev0[p, 0], scale=np.sqrt(meandev0[p, 1]))
    plt.plot(x, n_pdf,color='blue', label=str('Normal distribution'),linewidth=1.4)

    dev0 = np.loadtxt('output4/dev0.dat')
    dev0 = dev0[:, p]
    g_pdf = gaussian_kde(dev0.T)
    pdf = g_pdf(x)
    plt.plot(x, pdf, color='green', label=str('Actual distribution'))

    plt.legend()
    plt.savefig("graphs4/pair_%d"%p, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()

