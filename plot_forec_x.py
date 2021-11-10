##################################################################
#--------------- Plotting routines for saved data ---------------
##################################################################
'''
Plotting routine: <plot_func_x>

Loads saved data in specific directories and produces plots as a function of x at a given assimilation time. To use, specify (1) dir_name, (2) combination of parameters ijk, (3) time level T = time_vec[ii], i.e., choose ii.

NOTE: Any changes to the outer loop parameters should be replicated here too.

NOTE: currently saves as .png files

CALL WITH "python plot_func_x.py <i> <j> <k> <l> <time>

Assumes only one RTPP value.
'''


# generic modules 
import os
import errno
import numpy as np
import matplotlib
import itertools
import importlib.util
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from crps_calc_fun import crps_calc

##################################################################
##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

L = config.L
outdir = config.outdir
loc = config.loc
add_inf = config.add_inf
rtpp = config.rtpp
rtps = config.rtps
Neq = config.Neq
dres = config.dres
Nk_fc = config.Nk_fc
Nmeas = config.Nmeas
Nforec = config.Nforec
assim_time = config.assim_time
n_d = config.n_d
ob_noise = config.ob_noise
Hc = config.Hc
Hr = config.Hr
n_ens = config.n_ens
obs_h_d = config.obs_h_d
obs_u_d = config.obs_u_d
if(Neq==4): obs_v_d = config.obs_v_d
obs_r_d = config.obs_r_d
n_obs_h = config.n_obs_h
n_obs_u = config.n_obs_u
if(Neq==4): n_obs_v = config.n_obs_v
n_obs_r = config.n_obs_r

n_job = int(sys.argv[2])-1
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
i = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

## 2. CHOOSE time: plot at assimilation cycle ii
ii = int(sys.argv[3])

## 3. CHOOSE lead time: plot forecast at lead time jj
jj = int(sys.argv[4])
##################################################################
# make fig directory (if it doesn't already exist)

dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]),
                                                 str(add_inf[j]), str(rtpp[k]),
                                                 str(rtps[l]))
figsdir = str(dirn+'/figs')

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

## load data
print('*** Loading saved data... ')
B = np.load(str(dirn+'/B.npy')) #topography
#X = np.load(str(dirn+'/X_array.npy')) # fc ensembles
Xforec = np.load(str(dirn+'/X_forec.npy')) #long-range forecast
X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensemble
Y_obs = np.load(str(outdir+'/Y_obs_2xres_1h.npy')) # obs
OI = np.load(str(dirn+'/OI.npy')) # OI

# print shape of data arrays to terminal (sanity check)
print(' Check array shapes...')
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
#print('X_array shape (n_d,n_ens,T)      : ', np.shape(X)) 
print('X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)) 
print('Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan)) 
print('Y_obs shape (p,T-1)    : ', np.shape(Y_obs)) 
print(' ')
##################################################################

# determine parameters from loaded arrays
Kk_fc = 1./Nk_fc
xc = np.linspace(Kk_fc/2,L-Kk_fc/2,Nk_fc) 
t_an = np.shape(Xan)[2]
time_vec = list(range(0,t_an))
print('time_vec = ', time_vec)
print(' ')
T = time_vec[ii]
lead_time = jj

print(' *** Plotting at time T level = ', T+lead_time)
print(' *** Assim. time: ', assim_time[T+lead_time])

# masks for locating model variables in state vector
if(Neq==3):
    h_mask = list(range(0,Nk_fc))
    hu_mask = list(range(Nk_fc,2*Nk_fc))
    hr_mask = list(range(2*Nk_fc,3*Nk_fc))
    h_obs_mask = list(range(0,n_obs_h))
    hu_obs_mask = list(range(n_obs_h,n_obs_h+n_obs_u))
    hr_obs_mask = list(range(n_obs_h+n_obs_u,n_obs_h+n_obs_u+n_obs_r))
if(Neq==4):
    h_mask = list(range(0,Nk_fc))
    hu_mask = list(range(Nk_fc,2*Nk_fc))
    hv_mask = list(range(2*Nk_fc,3*Nk_fc))
    hr_mask = list(range(3*Nk_fc,4*Nk_fc)) 
    h_obs_mask = list(range(0,n_obs_h))
    hu_obs_mask = list(range(n_obs_h,n_obs_h+n_obs_u))
    hv_obs_mask = list(range(n_obs_h+n_obs_u,n_obs_h+n_obs_u+n_obs_v))
    hr_obs_mask = list(range(n_obs_h+n_obs_u+n_obs_v,n_obs_h+n_obs_u+n_obs_v+n_obs_r))

# masks for locating obs locations
row_vec_h = list(range(obs_h_d, Nk_fc+1, obs_h_d))
row_vec_u = list(range(Nk_fc+obs_u_d, 2*Nk_fc+1, obs_u_d))
if(Neq==3):
    row_vec_r = list(range(2*Nk_fc+obs_r_d, 3*Nk_fc+1, obs_r_d))
    row_vec = np.array(row_vec_h+row_vec_u+row_vec_r)
if(Neq==4):
    row_vec_v = list(range(2*Nk_fc+obs_r_d, 3*Nk_fc+1, obs_r_d)) 
    row_vec_r = list(range(3*Nk_fc+obs_r_d, 4*Nk_fc+1, obs_r_d))
    row_vec = np.array(row_vec_h+row_vec_u+row_vec_v+row_vec_r)

##################################################################

Xforbar = np.empty(np.shape(Xforec))

### CALCULATING ERRORS AT DIFFERENT LEAD TIMES ###
Xforbar[:,:,T,lead_time] = np.repeat(Xforec[:,:,T,lead_time].mean(axis=1), n_ens).reshape(n_d, n_ens)

##################################################################
frac = 0.15 # alpha value for translucent plotting
##################################################################
### 6 panel subplot for evolution of 3 vars: fc and an
##################################################################

fig, axes = plt.subplots(Neq, 1, figsize=(8,10))
#plt.suptitle("Ensemble trajectories (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)
axes[0].plot(xc, Xforec[h_mask,1:,T,lead_time]+B.reshape(len(xc),1), 'b',alpha=frac)
axes[0].plot(xc, Xforec[h_mask,0,T,lead_time]+B, 'b',alpha=frac,label="fc. ens.")
axes[0].plot(xc, Xforbar[h_mask,0,T,lead_time]+B, 'r',label="Ens. mean")
axes[0].plot(xc, X_tr[h_mask,0,T+lead_time]+B, 'g',label="Truth")
axes[0].errorbar(xc[row_vec[h_obs_mask]-1], Y_obs[h_obs_mask, T+lead_time] + B[row_vec[h_obs_mask]-1],
                   ob_noise[0], fmt='go',linewidth=2.0,label="Obs.")
axes[0].plot(xc,Hc*np.ones(len(xc)),'k:')
axes[0].plot(xc,Hr*np.ones(len(xc)),'k:')
axes[0].plot(xc, B, 'k', linewidth=2.0)
axes[0].set_ylim([0,0.1+np.max(X_tr[h_mask,:,T+lead_time]+B)])
axes[0].set_ylabel('$h(x)+b(x)$',fontsize=18)
axes[0].legend(loc = 1)

axes[1].plot(xc, Xforec[hu_mask,:,T,lead_time], 'b',alpha=frac)
axes[1].plot(xc, Xforbar[hu_mask,0,T,lead_time], 'r')
axes[1].plot(xc, X_tr[hu_mask,:,T+lead_time], 'g')
axes[1].plot(xc[row_vec[hu_obs_mask]-Nk_fc-1], Y_obs[hu_obs_mask,T+lead_time], 'go',linewidth=2.0)
axes[1].errorbar(xc[row_vec[hu_obs_mask]-Nk_fc-1], Y_obs[hu_obs_mask,T+lead_time], ob_noise[1],
                   fmt='go',linewidth=2.0)
axes[1].set_ylabel('$u(x)$',fontsize=18)

if(Neq==3):
    axes[2].plot(xc, Xforec[hr_mask,:,T,lead_time], 'b',alpha=frac)
    axes[2].plot(xc, Xforbar[hr_mask,0,T,lead_time], 'r')
    axes[2].plot(xc, X_tr[hr_mask,:,T+lead_time], 'g')
    axes[2].errorbar(xc[row_vec[hr_obs_mask]-2*Nk_fc-1], Y_obs[hr_obs_mask,T+lead_time], ob_noise[2], fmt='go',
                       linewidth=2.0)
    axes[2].plot(xc,np.zeros(len(xc)),'k')
    axes[2].set_ylabel('$r(x)$',fontsize=18)
    axes[2].set_ylim([-0.025,0.02+np.max(X_tr[hr_mask,0,T+lead_time])])
    axes[2].set_xlabel('$x$',fontsize=18)
if(Neq==4):
    axes[2].plot(xc, Xforec[hv_mask,:,T,lead_time], 'b',alpha=frac)
    axes[2].plot(xc, Xforbar[hv_mask,0,T,lead_time], 'r')
    axes[2].plot(xc, X_tr[hv_mask,:,T+lead_time], 'g')
    axes[2].errorbar(xc[row_vec[hv_obs_mask]-2*Nk_fc-1], Y_obs[hv_obs_mask,T+lead_time], ob_noise[2], fmt='go',
                       linewidth=2.0)
    axes[2].plot(xc,np.zeros(len(xc)),'k')
    axes[2].set_ylabel('$v(x)$',fontsize=18)
    axes[3].plot(xc, Xforec[hr_mask,:,T,lead_time], 'b',alpha=frac)
    axes[3].plot(xc, Xforbar[hr_mask,0,T,lead_time], 'r')
    axes[3].plot(xc, X_tr[hr_mask,:,T+lead_time], 'g')
    axes[3].errorbar(xc[row_vec[hr_obs_mask]-3*Nk_fc-1], Y_obs[hr_obs_mask,T+lead_time], ob_noise[3], fmt='go',
                       linewidth=2.0)
    axes[3].plot(xc,np.zeros(len(xc)),'k')
    axes[3].set_ylabel('$r(x)$',fontsize=18)
    axes[3].set_ylim([-0.025,0.02+np.max(X_tr[hr_mask,0,T+lead_time])])
    axes[3].set_xlabel('$x$',fontsize=18)

name_f = "/T"+str(T)+"_assim_lead+"+str(lead_time)+".png"
f_name_f = str(figsdir+name_f)
plt.savefig(f_name_f)
print(' ')
print(' *** %s at time level %d saved to %s' %(name_f,T,figsdir))

