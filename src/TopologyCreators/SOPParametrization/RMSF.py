
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import MDAnalysis as mda
import MDAnalysis.analysis.align

u_ref = mda.Universe('ubq_FA.pdb')
u_aa = mda.Universe('ubq_FA.pdb', 'ubq_FA.dcd')
u_cg_0 = mda.Universe('ubq_FA.pdb', 'ubq_CG_0.dcd')
u_cg_1 = mda.Universe('ubq_FA.pdb', 'ubq_CG_1024.dcd')

MDAnalysis.analysis.align.AlignTraj(u_aa, u_ref, select='all', filename = 'ubq_FA_aligned.xtc').run()
MDAnalysis.analysis.align.AlignTraj(u_cg_0, u_ref, select='all', filename = 'ubq_CG_0_aligned.xtc').run()
MDAnalysis.analysis.align.AlignTraj(u_cg_1, u_ref, select='all', filename = 'ubq_CG_1024_aligned.xtc').run()

# aligned trajectory
u_aa = mda.Universe('ubq_FA.pdb', 'ubq_FA_aligned.xtc')
u_cg_0 = mda.Universe('ubq_FA.pdb', 'ubq_CG_0_aligned.xtc')
u_cg_1 = mda.Universe('ubq_FA.pdb', 'ubq_CG_1024_aligned.xtc')

from MDAnalysis.analysis import rms
u_aa_sel = u_aa.select_atoms("all")
u_cg_0_sel = u_cg_0.select_atoms("all")
u_cg_1_sel = u_cg_1.select_atoms("all")

u_aa_rmsf = rms.RMSF(u_aa_sel)
u_aa_rmsf.run()

u_cg_0_rmsf = rms.RMSF(u_cg_0_sel)
u_cg_0_rmsf.run()

u_cg_1_rmsf = rms.RMSF(u_cg_1_sel)
u_cg_1_rmsf.run()


# In[8]:


import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

params = {'legend.fontsize': '20',
          'figure.figsize': (12, 8),
         'axes.labelsize': '22',
         'axes.titlesize':'22',
         'xtick.labelsize':'22',
         'ytick.labelsize':'22'}
matplotlib.rcParams.update(params)

ax.plot(u_aa_sel.resids, u_aa_rmsf.rmsf, lw = 3, color = 'k', label = 'Full-atomic')
ax.plot(u_aa_sel.resids, u_cg_0_rmsf.rmsf,  lw = 3, color = 'r', label = 'Coarse-grained, before parametrization')
ax.plot(u_aa_sel.resids, u_cg_1_rmsf.rmsf,  lw = 3, color = 'b', label = 'Coarse-grained, after parametrization')
ax.legend(loc="best")
ax.set_xlabel("Residue")
ax.set_ylabel(r"RMSF, angstr")
plt.savefig("RMSF", dpi=300)
plt.show()


# In[5]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(10,10))

sns.set(color_codes=True)

ax = fig.add_subplot(211)
sns.regplot(x=u_aa_rmsf.rmsf, y=u_cg_0_rmsf.rmsf, color = 'r')

corr_coef = np.corrcoef(u_aa_rmsf.rmsf, u_cg_0_rmsf.rmsf)
ax.set_xlabel("Pearson's correlation = " + str(round(corr_coef[0, 1], 3)))

ax = fig.add_subplot(212)
sns.regplot(x=u_aa_rmsf.rmsf, y=u_cg_1_rmsf.rmsf, color = 'b')

corr_coef = np.corrcoef(u_aa_rmsf.rmsf, u_cg_1_rmsf.rmsf)
ax.set_xlabel("Pearson's correlation = " + str(round(corr_coef[0, 1], 3)))

plt.show()

