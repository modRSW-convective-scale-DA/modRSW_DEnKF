##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno
import sys
import importlib.util

# HANDLE WARNINGS AS ERRORS
##################################################################
import warnings
warnings.filterwarnings("error")

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

#from parameters import *
from f_modRSW import make_grid
from f_enkf_modRSW import generate_truth
from init_cond_modRSW import init_cond_topog_cos
from create_readme import create_readme

#################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE                     #
#################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
Nk_tr = config.Nk_tr
Nk_fc = config.Nk_fc
L = config.L
cfl_tr = config.cfl_tr
Neq = config.Neq
ic = config.ic
H0 = config.H0
A = config.A
V = config.V
Hc = config.Hc
Hr = config.Hr
Ro = config.Ro
cc2 = config.cc2
beta = config.beta
alpha2 = config.alpha2
g = config.g
tmax = config.tmax
assim_time = config.assim_time
Nmeas = config.Nmeas
Nforec = config.Nforec
n_obs = config.n_obs
dres = config.dres
n_d = config.n_d
ob_noise = config.ob_noise
obs_h_d = config.obs_h_d
obs_u_d = config.obs_u_d
if(Neq==4): obs_v_d = config.obs_v_d
obs_r_d = config.obs_r_d
n_obs_h = config.n_obs_h
n_obs_u = config.n_obs_u
if(Neq==4): n_obs_v = config.n_obs_v
n_obs_r = config.n_obs_r
U_relax = config.U_relax
tau_rel = config.tau_rel
h_obs_mask = config.h_obs_mask
hu_obs_mask = config.hu_obs_mask
hr_obs_mask = config.hr_obs_mask
if(Neq==4): hv_obs_mask = config.hv_obs_mask

#################################################################
# create directory for output
#################################################################
#check if dir exixts, if not make it
try:
    os.makedirs(outdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise


##################################################################    
# Mesh generation and IC for truth 
##################################################################
tr_grid =  make_grid(Nk_tr,L) # truth
Kk_tr = tr_grid[0]
x_tr = tr_grid[1]
xc_tr = tr_grid[2]

### Truth ic
U0_tr, B_tr = ic(x_tr,Nk_tr,Neq,H0,L,A,V)
np.save(str(outdir+'/B_tr'),B_tr) #save topog for plotting

U_tr_array = np.empty([Neq,Nk_tr,Nmeas+Nforec+1])
U_tr_array[:,:,0] = U0_tr

f_path_name = str(outdir+'/U_tr_array_2xres_1h.npy')
f_obs_name = str(outdir+'/Y_obs_2xres_1h.npy')
f_H_name = str(outdir+'/H.npy')

U_rel_tr = U_relax(Neq,Nk_tr,L,V,xc_tr,U0_tr)

try:
    print(' *** Loading truth trajectory... *** ')
    U_tr_array = np.load(f_path_name)
except:
    print(' *** Generating truth trajectory... *** ')
    U_tr_array = generate_truth(U_tr_array, U_rel_tr, Neq, Nk_tr, tr_grid, cfl_tr, assim_time, tmax, f_path_name, Hc, Hr, cc2, beta, alpha2, g, Ro, tau_rel)

##################################################################    
# Pseudo-observations
##################################################################

print('Total no. of obs. =', n_obs)

# Sample the truth trajectory on the forecast grid.
U_tmp = np.copy(U_tr_array[:, 0::dres, :])

# for assimilation, work with [h,u,r]
U_tmp[1:,:] = U_tmp[1:,:]/U_tmp[0,:]

try:
    H = np.load(f_H_name)
except:
    # observation operator
    H = np.zeros([n_obs, n_d])
    row_vec_h = list(range(obs_h_d, Nk_fc+1, obs_h_d))
    row_vec_u = list(range(Nk_fc+obs_u_d, 2*Nk_fc+1, obs_u_d))
    if(Neq==3):
        row_vec_r = list(range(2*Nk_fc+obs_r_d, 3*Nk_fc+1, obs_r_d))
        row_vec = row_vec_h+row_vec_u+row_vec_r
    if(Neq==4): 
        row_vec_v = list(range(2*Nk_fc+obs_v_d, 3*Nk_fc+1, obs_v_d))
        row_vec_r = list(range(3*Nk_fc+obs_r_d, 4*Nk_fc+1, obs_r_d))
        row_vec = row_vec_h+row_vec_u+row_vec_v+row_vec_r
    #row_vec = range(o_d, n_d+1, o_d)
    for i in range(0, n_obs):
        H[i, row_vec[i]-1] = 1
    np.save(f_H_name,H)

try:
    Y_obs = np.load(f_obs_name)
except:
    # create imperfect observations by adding the same observation noise
    # to each member's perfect observation. N.B. The observation time index
    # is shifted by one cycle relative to the truth trajectory because
    # no observations are assimilated at the initial condition time.
    Y_obs = np.empty([n_obs, Nmeas])
    if(Neq==3): ob_noise = np.repeat(ob_noise, [n_obs_h,n_obs_u,n_obs_r])
    if(Neq==4): ob_noise = np.repeat(ob_noise, [n_obs_h,n_obs_u,n_obs_v,n_obs_r])
    for T in range(np.size(assim_time)-1):
        X_tr = U_tmp[:, :, T+1].flatten()
        X_tr = X_tr.T
        Y_mod = np.dot(H, X_tr)
        obs_pert = ob_noise * np.random.randn(n_obs)
        Y_obs[:, T] = Y_mod.flatten() + obs_pert

        # Reset pseudo-observations with negative h or r to zero.
        if(n_obs_h!=0 and n_obs_r!=0): mask = np.append(h_obs_mask[np.array(Y_obs[h_obs_mask, T] < 0.0)], hr_obs_mask[np.array(Y_obs[hr_obs_mask, T] < 0.0)])
        elif(n_obs_r!=0 and n_obs_h==0): mask = hr_obs_mask[np.array(Y_obs[hr_obs_mask, T] < 0.0)]
        elif(n_obs_r==0 and n_obs_h!=0): mask = h_obs_mask[np.array(Y_obs[h_obs_mask, T] < 0.0)]
        Y_obs[mask, T] = 0.0
    np.save(f_obs_name, Y_obs)


#### END OF THE PROGRAM ####
