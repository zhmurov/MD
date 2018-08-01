import numpy as np
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# TODO ========
iteration = [1, 2, 8, 16, 32, 64, 128, 256, 512]
Npairs = 32
# =============

pair = range(Npairs)

cmap = plt.get_cmap('Greens')
colors = []
for i in np.linspace(0.2,1,len(iteration)):
    colors.append(cmap(i))
    
print colors[-2]

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
matplotlib.rcParams.update(params)

i = iteration[0]


# TODO axis X bounds
#v_min = np.amin(dev)
#v_max = np.amax(dev)
#v_max=v_max/2
v_min = 2
v_max = 10

x = np.linspace(v_min, v_max, 512)

for p in pair:
    plt.ylabel('Probability density function',fontsize=20)
    plt.xlabel('Contact distance, angstr',fontsize=20)

    dev0 = np.loadtxt('output2/dev0.dat')
    dev0 = dev0[:,p]
    g_pdf = gaussian_kde(dev0.T)
    pdf = g_pdf(x)
    plt.plot(x, pdf, color=(1.0, 0.0, 0.0, 1.0), label=str('Full-atomic'), linewidth=1.4)

    col = 0;    #color
    for i in iteration:
        dev = np.loadtxt('output2/dev%d.dat'%i)
        dev = dev[:, p]
        g_pdf = gaussian_kde(dev.T)
        pdf = g_pdf(x)
        plt.plot(x, pdf, color=colors[col], label=str('Iteration %d'%i), linewidth=1.4)
        col = col+1

    plt.legend()
    plt.savefig("graphs2/pair_%d"%p, dpi=300)
    plt.clf()
    plt.cla()
    plt.close()
