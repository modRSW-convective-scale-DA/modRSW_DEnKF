'''
    A few checks on characteristics of the nature run: plot all trajectories, check height and  rain extremes
    '''


import numpy as np
import os
import matplotlib.pyplot as plt
from pylab import *

dirname = '/kent_enkf/2xres_1h'

cwd = "/nobackup/mmlca"
dirn = str(cwd+dirname)

U_tr = np.load(str(dirn+'/U_tr_array_2xres_1h.npy'))

print('Shape of truth array is:', np.shape(U_tr))
print('Maximum height reached is:', np.max(U_tr[0,:,:]))
print('Minimum height reached is:', np.min(U_tr[0,:,:]))

print('Max rain is:', np.max(U_tr[2,:,:]/U_tr[0,:,:]))
print('Min rain is:', np.min(U_tr[2,:,:]/U_tr[0,:,:]))

figure()
plot(U_tr[0,:,:])
plot(U_tr[0,:,-1],linewidth = 3)
show()
