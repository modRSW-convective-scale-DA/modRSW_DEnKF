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
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import itertools

#from parameters import *
from crps_calc_fun import crps_calc

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
h_obs_mask = config.h_obs_mask
hu_obs_mask = config.hu_obs_mask
hr_obs_mask = config.hr_obs_mask
if(Neq==4): hv_obs_mask = config.hv_obs_mask

## 1. CHOOSE ijkl. E.g., for test_enkf1111/ [i,j,k,l] = [0,0,0,0]
n_job = int(sys.argv[2])-1
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
i = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

## 2. CHOOSE time: plot at assimilation cycle ii
ii = int(sys.argv[3])
##################################################################
# make fig directory (if it doesn't already exist)
dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]),
                                                 str(add_inf[j]), str(rtpp[k]),
                                                 str(rtps[l]))# make fig directory (if it doesn't already exist)

figsdir = str(dirn+'/figs')

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

## load data
print('*** Loading saved data... ')
B = np.load(str(dirn+'/B.npy')) #topography
X = np.load(str(dirn+'/X_array.npy')) # fc ensembles
X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensemble
Y_obs = np.load(str(outdir+'/Y_obs_2xres_1h.npy')) # obs
OI = np.load(str(dirn+'/OI.npy')) # OI

# print shape of data arrays to terminal (sanity check)
print(' Check array shapes...')
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('X_array shape (n_d,n_ens,T)      : ', np.shape(X)) 
print('X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)) 
print('Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan)) 
print('Y_obs shape (p,T-1)    : ', np.shape(Y_obs)) 
print(' ')
##################################################################

# determine parameters from loaded arrays
Kk_fc = 1./Nk_fc
xc = np.linspace(Kk_fc/2,L-Kk_fc/2,Nk_fc)
t_an = np.shape(X)[2]
time_vec = list(range(0,t_an))
print('time_vec = ', time_vec)
print(' ')
T = time_vec[ii]

print(' *** Plotting at time T level = ', T)
print(' *** Assim. time: ', assim_time[T])

# masks for locating model variables in state vector
if(Neq==3):
    h_mask = list(range(0,Nk_fc))
    hu_mask = list(range(Nk_fc,2*Nk_fc))
    hr_mask = list(range(2*Nk_fc,3*Nk_fc))

if(Neq==4):
    h_mask = list(range(0,Nk_fc))
    hu_mask = list(range(Nk_fc,2*Nk_fc))
    hv_mask = list(range(2*Nk_fc,3*Nk_fc))
    hr_mask = list(range(3*Nk_fc,4*Nk_fc))

# masks for locating obs locations
row_vec_h = list(range(obs_h_d, Nk_fc+1, obs_h_d))
row_vec_u = list(range(Nk_fc+obs_u_d, 2*Nk_fc+1, obs_u_d))
if(Neq==3):
    row_vec_r = list(range(2*Nk_fc+obs_r_d, 3*Nk_fc+1, obs_r_d))
    row_vec = np.array(row_vec_h+row_vec_u+row_vec_r)
if(Neq==4):
    row_vec_v = list(range(2*Nk_fc+obs_v_d, 3*Nk_fc+1, obs_v_d))
    row_vec_r = list(range(3*Nk_fc+obs_r_d, 4*Nk_fc+1, obs_r_d))
    row_vec = np.array(row_vec_h+row_vec_u+row_vec_v+row_vec_r)

##################################################################

# compute means and deviations
Xbar = np.empty(np.shape(X))
Xdev = np.empty(np.shape(X))
Xanbar = np.empty(np.shape(X))
Xandev = np.empty(np.shape(X))
Xdev_tr = np.empty(np.shape(X))
Xandev_tr = np.empty(np.shape(X))

#ONE = np.ones([n_ens,n_ens])
#ONE = ONE/n_ens # NxN array with elements equal to 1/N
for ii in time_vec:
#    Xbar[:,:,ii] = np.dot(X[:,:,ii],ONE) # fc mean
    Xbar[:,:,ii] = np.repeat(X[:,:,ii].mean(axis=1), n_ens).reshape(n_d, n_ens)
    Xdev[:,:,ii] = X[:,:,ii] - Xbar[:,:,ii] # fc deviations from mean
    Xdev_tr[:,:,ii] = X[:,:,ii] - X_tr[:,:,ii] # fc deviations from truth
#    Xanbar[:,:,ii] = np.dot(Xan[:,:,ii],ONE) # an mean
    Xanbar[:,:,ii] = np.repeat(Xan[:,:,ii].mean(axis=1), n_ens).reshape(n_d, n_ens)
    Xandev[:,:,ii] = Xan[:,:,ii] - Xanbar[:,:,ii] # an deviations from mean
    Xandev_tr[:,:,ii] = Xan[:,:,ii] - X_tr[:,:,ii] # an deviations from truth

##################################################################
frac = 0.15 # alpha value for translucent plotting
##################################################################
### 6 panel subplot for evolution of 3 vars: fc and an
##################################################################

fig, axes = plt.subplots(Neq, 2, figsize=(15,10))
#plt.suptitle("Ensemble trajectories (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)

axes[0,0].plot(xc, X[h_mask,1:,T]+B.reshape(len(xc),1), 'b',alpha=frac)
axes[0,0].plot(xc, X[h_mask,0,T]+B, 'b',alpha=frac,label="fc. ens.")
axes[0,0].plot(xc, Xbar[h_mask,0,T]+B, 'r',label="Ens. mean")
axes[0,0].plot(xc, X_tr[h_mask,0,T]+B, 'g',label="Truth")
axes[0,0].errorbar(xc[row_vec[h_obs_mask]-1], Y_obs[h_obs_mask, T] + B[row_vec[h_obs_mask]-1],
                   ob_noise[0], fmt='go',linewidth=2.0,label="Obs.")
axes[0,0].plot(xc,Hc*np.ones(len(xc)),'k:')
axes[0,0].plot(xc,Hr*np.ones(len(xc)),'k:')
axes[0,0].plot(xc, B, 'k', linewidth=2.0)
axes[0,0].set_ylim([0,0.1+np.max(X_tr[h_mask,:,T]+B)])
axes[0,0].set_ylabel('$h(x)+b(x)$',fontsize=18)
axes[0,0].legend(loc = 1)


axes[0,1].plot(xc, Xan[h_mask,1:,T]+B.reshape(len(xc),1), 'b',alpha=frac)
axes[0,1].plot(xc, Xan[h_mask,0,T]+B, 'b',alpha=frac,label="an. ens.")
axes[0,1].plot(xc, Xanbar[h_mask,0,T]+B, 'c',linewidth=2.0,label="Analysis")
axes[0,1].plot(xc, X_tr[h_mask,0,T]+B, 'g',label="Truth")
axes[0,1].errorbar(xc[row_vec[h_obs_mask]-1], Y_obs[h_obs_mask, T] + B[row_vec[h_obs_mask]-1],
                   ob_noise[0], fmt='go',linewidth=2.0,label="Obs.")
axes[0,1].plot(xc,Hc*np.ones(len(xc)),'k:')
axes[0,1].plot(xc,Hr*np.ones(len(xc)),'k:')
axes[0,1].plot(xc, B, 'k', linewidth=2.0)
axes[0,1].set_ylim([0,0.1+np.max(X_tr[h_mask,:,T]+B)])
axes[0,1].legend(loc = 1)

axes[1,0].plot(xc, X[hu_mask,:,T], 'b',alpha=frac)
axes[1,0].plot(xc, Xbar[hu_mask,0,T], 'r')
axes[1,0].plot(xc, X_tr[hu_mask,:,T], 'g')
axes[1,0].plot(xc[row_vec[hu_obs_mask]-Nk_fc-1], Y_obs[hu_obs_mask,T], 'go',linewidth=2.0)
axes[1,0].errorbar(xc[row_vec[hu_obs_mask]-Nk_fc-1], Y_obs[hu_obs_mask,T], ob_noise[1],
                   fmt='go',linewidth=2.0)
axes[1,0].set_ylabel('$u(x)$',fontsize=18)

axes[1,1].plot(xc, Xan[hu_mask,:,T], 'b',alpha=frac)
axes[1,1].plot(xc, Xanbar[hu_mask,0,T], 'c',linewidth=2.0)
axes[1,1].plot(xc, X_tr[hu_mask,:,T], 'g')
axes[1,1].errorbar(xc[row_vec[hu_obs_mask]-Nk_fc-1], Y_obs[hu_obs_mask,T], ob_noise[1],
                   fmt='go',linewidth=2.0)

if(Neq==3):
    axes[2,0].plot(xc, X[hr_mask,:,T], 'b',alpha=frac)
    axes[2,0].plot(xc, Xbar[hr_mask,0,T], 'r')
    axes[2,0].plot(xc, X_tr[hr_mask,:,T], 'g')
    axes[2,0].errorbar(xc[row_vec[hr_obs_mask]-2*Nk_fc-1], Y_obs[hr_obs_mask,T], ob_noise[2], fmt='go',
                   linewidth=2.0)
    axes[2,0].plot(xc,np.zeros(len(xc)),'k')
    axes[2,0].set_ylabel('$r(x)$',fontsize=18)
    axes[2,0].set_ylim([-0.025,0.02+np.max(X_tr[hr_mask,0,T])])
    axes[2,0].set_xlabel('$x$',fontsize=18)

    axes[2,1].plot(xc, Xan[hr_mask,:,T], 'b',alpha=frac)
    axes[2,1].plot(xc, Xanbar[hr_mask,0,T], 'c',linewidth=2.0)
    axes[2,1].plot(xc, X_tr[hr_mask,:,T], 'g')
    axes[2,1].errorbar(xc[row_vec[hr_obs_mask]-2*Nk_fc-1], Y_obs[hr_obs_mask,T], ob_noise[2], fmt='go',
                   linewidth=2.0)
    axes[2,1].plot(xc,np.zeros(len(xc)),'k')
    axes[2,1].set_ylim([-0.025,0.02+np.max(X_tr[hr_mask,0,T])])
    axes[2,1].set_xlabel('$x$',fontsize=18)
if(Neq==4):
    axes[2,0].plot(xc, X[hv_mask,:,T], 'b',alpha=frac)
    axes[2,0].plot(xc, Xbar[hv_mask,0,T], 'r')
    axes[2,0].plot(xc, X_tr[hv_mask,:,T], 'g')
    axes[2,0].errorbar(xc[row_vec[hv_obs_mask]-2*Nk_fc-1], Y_obs[hv_obs_mask,T], ob_noise[2], fmt='go',
                   linewidth=2.0)
    axes[2,0].set_ylabel('$v(x)$',fontsize=18)

    axes[2,1].plot(xc, Xan[hv_mask,:,T], 'b',alpha=frac)
    axes[2,1].plot(xc, Xanbar[hv_mask,0,T], 'c',linewidth=2.0)
    axes[2,1].plot(xc, X_tr[hv_mask,:,T], 'g')
    axes[2,1].errorbar(xc[row_vec[hv_obs_mask]-2*Nk_fc-1], Y_obs[hv_obs_mask,T], ob_noise[2], fmt='go',
                   linewidth=2.0)
    axes[2,1].plot(xc,np.zeros(len(xc)),'k')
    axes[3,0].plot(xc, X[hr_mask,:,T], 'b',alpha=frac)
    axes[3,0].plot(xc, Xbar[hr_mask,0,T], 'r')
    axes[3,0].plot(xc, X_tr[hr_mask,:,T], 'g')
    axes[3,0].errorbar(xc[row_vec[hr_obs_mask]-3*Nk_fc-1], Y_obs[hr_obs_mask,T], ob_noise[3], fmt='go',
                   linewidth=2.0)
    axes[3,0].plot(xc,np.zeros(len(xc)),'k')
    axes[3,0].set_ylabel('$r(x)$',fontsize=18)
    axes[3,0].set_ylim([-0.025,0.02+np.max(X_tr[hr_mask,0,T])])
    axes[3,0].set_xlabel('$x$',fontsize=18)

    axes[3,1].plot(xc, Xan[hr_mask,:,T], 'b',alpha=frac)
    axes[3,1].plot(xc, Xanbar[hr_mask,0,T], 'c',linewidth=2.0)
    axes[3,1].plot(xc, X_tr[hr_mask,:,T], 'g')
    axes[3,1].errorbar(xc[row_vec[hr_obs_mask]-3*Nk_fc-1], Y_obs[hr_obs_mask,T], ob_noise[3], fmt='go',
                   linewidth=2.0)
    axes[3,1].plot(xc,np.zeros(len(xc)),'k')
    axes[3,1].set_ylim([-0.025,0.02+np.max(X_tr[hr_mask,0,T])])
    axes[3,1].set_xlabel('$x$',fontsize=18)

name_f = "/T%d_assim.png" %T
f_name_f = str(figsdir+name_f)
plt.savefig(f_name_f)
print(' ')
print(' *** %s at time level %d saved to %s' %(name_f,T,figsdir))

##################################################################
###                       ERRORS                              ####
##################################################################

## ANALYSIS
an_err = Xanbar[:,0,T] - X_tr[:,0,T] # an_err = analysis ens. mean - truth
an_err2 = an_err**2
# domain-averaged mean errors
an_ME_h = an_err[h_mask].mean() 
an_ME_hu = an_err[hu_mask].mean()
if(Neq==4): an_ME_hv = an_err[hv_mask].mean()
an_ME_hr = an_err[hr_mask].mean()
# domain-averaged absolute errors
an_absME_h = np.absolute(an_err[h_mask])
an_absME_hu = np.absolute(an_err[hu_mask])
if(Neq==4): an_absME_hv = np.absolute(an_err[hv_mask])
an_absME_hr = np.absolute(an_err[hr_mask])

# cov matrix
Pa = np.dot(Xandev[:,:,T],np.transpose(Xandev[:,:,T]))
Pa = Pa/(n_ens - 1) # analysis covariance matrix
var_an = np.diag(Pa)

Pa_tr = np.dot(Xandev_tr[:,:,T],np.transpose(Xandev_tr[:,:,T]))
Pa_tr = Pa_tr/(n_ens - 1) # fc covariance matrix w.r.t truth
var_ant = np.diag(Pa_tr)

## FORECAST
fc_err = Xbar[:,0,T] - X_tr[:,0,T] # fc_err = ens. mean - truth
fc_err2 = fc_err**2
# domain-averaged mean errors
fc_ME_h = fc_err[h_mask].mean()
fc_ME_hu = fc_err[hu_mask].mean()
if(Neq==4): fc_ME_hv = fc_err[hv_mask].mean()
fc_ME_hr = fc_err[hr_mask].mean()
# domain-averaged absolute errors
fc_absME_h = np.absolute(fc_err[h_mask])
fc_absME_hu = np.absolute(fc_err[hu_mask])
if(Neq==4): fc_absME_hv = np.absolute(fc_err[hv_mask])
fc_absME_hr = np.absolute(fc_err[hr_mask])

# cov matrix
Pf = np.dot(Xdev[:,:,T],np.transpose(Xdev[:,:,T]))
Pf = Pf/(n_ens - 1) # fc covariance matrix
var_fc = np.diag(Pf)

Pf_tr = np.dot(Xdev_tr[:,:,T],np.transpose(Xdev_tr[:,:,T]))
Pf_tr = Pf_tr/(n_ens - 1) # fc covariance matrix w.r.t. truth
var_fct = np.diag(Pf_tr)
'''
# fc/an
ME_ratio_h = np.sqrt(fc_err2[h_mask])/np.sqrt(an_err2[h_mask])
ME_ratio_hu = np.sqrt(fc_err2[hu_mask])/np.sqrt(an_err2[hu_mask])
ME_ratio_hr = np.sqrt(fc_err2[hr_mask])/np.sqrt(an_err2[hr_mask])
# fc - an
ME_diff_h = np.sqrt(fc_err2[h_mask])-np.sqrt(an_err2[h_mask])
ME_diff_hu = np.sqrt(fc_err2[hu_mask])-np.sqrt(an_err2[hu_mask])
ME_diff_hr = np.sqrt(fc_err2[hr_mask])-np.sqrt(an_err2[hr_mask])
'''
##################################################################

# fontsize
ft = 16

# position text on plot
pl_h = np.max([np.sqrt(var_fc[h_mask]),fc_absME_h])
pl_hu = np.max([np.sqrt(var_fc[hu_mask]),fc_absME_hu])
pl_hr = np.max([np.sqrt(var_fc[hr_mask]),fc_absME_hr])
if(Neq==4): pl_hv = np.max([np.sqrt(var_fc[hv_mask]),fc_absME_hv])

# domain-averaged errors
an_spr_h = np.mean(np.sqrt(var_an[h_mask]))
an_rmse_h = np.mean(np.sqrt(var_ant[h_mask]))
fc_spr_h = np.mean(np.sqrt(var_fc[h_mask]))
fc_rmse_h = np.mean(np.sqrt(var_fct[h_mask]))

an_spr_hu = np.mean(np.sqrt(var_an[hu_mask]))
an_rmse_hu = np.mean(np.sqrt(var_ant[hu_mask]))
fc_spr_hu = np.mean(np.sqrt(var_fc[hu_mask]))
fc_rmse_hu = np.mean(np.sqrt(var_fct[hu_mask]))

an_spr_hr = np.mean(np.sqrt(var_an[hr_mask]))
an_rmse_hr = np.mean(np.sqrt(var_ant[hr_mask]))
fc_spr_hr = np.mean(np.sqrt(var_fc[hr_mask]))
fc_rmse_hr = np.mean(np.sqrt(var_fct[hr_mask]))

if(Neq==4):
    an_spr_hv = np.mean(np.sqrt(var_an[hv_mask]))
    an_rmse_hv = np.mean(np.sqrt(var_ant[hv_mask]))
    fc_spr_hv = np.mean(np.sqrt(var_fc[hv_mask]))
    fc_rmse_hv = np.mean(np.sqrt(var_fct[hv_mask]))


##################################################################
### 6 panel subplot: comparing spread and error for fc and an
##################################################################

fig, axes = plt.subplots(Neq, 2, figsize=(12,12))

axes[0,0].plot(xc, np.sqrt(var_fc[h_mask]),'r',label='fc spread') # spread
axes[0,0].plot(xc, fc_absME_h,'r--',label='fc err') # rmse
axes[0,0].plot(xc, np.sqrt(var_an[h_mask]),'b',label='an spread')
axes[0,0].plot(xc, an_absME_h,'b--',label='an err')
axes[0,0].set_ylabel('$h(x)$',fontsize=18)
axes[0,0].text(0.025, 1.2*pl_h, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_h,np.mean(an_absME_h)), fontsize=ft, color='b')
axes[0,0].text(0.025, 1.1*pl_h, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_h,np.mean(fc_absME_h)), fontsize=ft, color='r')
axes[0,0].set_ylim([0,1.3*pl_h])

axes[1,0].plot(xc, np.sqrt(var_fc[hu_mask]), 'r')
axes[1,0].plot(xc, fc_absME_hu, 'r--')
axes[1,0].plot(xc, np.sqrt(var_an[hu_mask]), 'b')
axes[1,0].plot(xc, an_absME_hu , 'b--')
axes[1,0].set_ylabel('$u(x)$',fontsize=18)
axes[1,0].text(0.025, 1.2*pl_hu, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_hu,an_absME_hu.mean()), fontsize=ft, color='b')
axes[1,0].text(0.025, 1.1*pl_hu, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_hu,fc_absME_hu.mean()), fontsize=ft, color='r')
axes[1,0].set_ylim([0,1.3*pl_hu])

if(Neq==3):
    axes[2,0].plot(xc, np.sqrt(var_fc[hr_mask]), 'r')
    axes[2,0].plot(xc, fc_absME_hr , 'r--')
    axes[2,0].plot(xc, np.sqrt(var_an[hr_mask]), 'b')
    axes[2,0].plot(xc, an_absME_hr , 'b--')
    axes[2,0].set_ylabel('$r(x)$',fontsize=18)
    axes[2,0].set_xlabel('$x$',fontsize=18)
    axes[2,0].text(0.025, 1.2*pl_hr, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_hr,an_absME_hr.mean() ), fontsize=ft, color='b')
    axes[2,0].text(0.025, 1.1*pl_hr, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_hr,fc_absME_hr.mean() ), fontsize=ft, color='r')
    axes[2,0].set_ylim([0,1.3*pl_hr])
if(Neq==4):
    axes[2,0].plot(xc, np.sqrt(var_fc[hv_mask]), 'r')
    axes[2,0].plot(xc, fc_absME_hv , 'r--')
    axes[2,0].plot(xc, np.sqrt(var_an[hv_mask]), 'b')
    axes[2,0].plot(xc, an_absME_hv , 'b--')
    axes[2,0].set_ylabel('$v(x)$',fontsize=18)
    axes[2,0].set_xlabel('$x$',fontsize=18)
    axes[2,0].text(0.025, 1.2*pl_hv, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_hv,an_absME_hv.mean() ), fontsize=ft, color='b')
    axes[2,0].text(0.025, 1.1*pl_hv, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_hv,fc_absME_hv.mean() ), fontsize=ft, color='r')
    axes[2,0].set_ylim([0,1.3*pl_hv])
    axes[3,0].plot(xc, np.sqrt(var_fc[hr_mask]), 'r')
    axes[3,0].plot(xc, fc_absME_hr , 'r--')
    axes[3,0].plot(xc, np.sqrt(var_an[hr_mask]), 'b')
    axes[3,0].plot(xc, an_absME_hr , 'b--')
    axes[3,0].set_ylabel('$r(x)$',fontsize=18)
    axes[3,0].set_xlabel('$x$',fontsize=18)
    axes[3,0].text(0.025, 1.2*pl_hr, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_hr,an_absME_hr.mean() ), fontsize=ft, color='b')
    axes[3,0].text(0.025, 1.1*pl_hr, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_hr,fc_absME_hr.mean() ), fontsize=ft, color='r')
    axes[3,0].set_ylim([0,1.3*pl_hr])

axes[0,1].plot(xc, fc_absME_h - np.sqrt(var_fc[h_mask]), 'r',label='fc: err  - spr')
axes[0,1].plot(xc, an_absME_h - np.sqrt(var_an[h_mask]), 'b',label='an: err - spr')
axes[0,1].plot(xc,np.zeros(len(xc)),'k:')
axes[0,1].legend(loc=0)

axes[1,1].plot(xc, fc_absME_hu - np.sqrt(var_fc[hu_mask]), 'r')
axes[1,1].plot(xc, an_absME_hu - np.sqrt(var_an[hu_mask]), 'b')
axes[1,1].plot(xc, np.zeros(len(xc)),'k:')

if(Neq==3):
    axes[2,1].plot(xc, fc_absME_hr - np.sqrt(var_fc[hr_mask]), 'r')
    axes[2,1].plot(xc, an_absME_hr - np.sqrt(var_an[hr_mask]), 'b')
    axes[2,1].plot(xc, np.zeros(len(xc)),'k:')
    axes[2,1].set_xlabel('$x$',fontsize=18)
if(Neq==4):
    axes[2,1].plot(xc, fc_absME_hv - np.sqrt(var_fc[hv_mask]), 'r')
    axes[2,1].plot(xc, an_absME_hv - np.sqrt(var_an[hv_mask]), 'b')
    axes[2,1].plot(xc, np.zeros(len(xc)),'k:')
    axes[2,1].set_xlabel('$x$',fontsize=18)
    axes[3,1].plot(xc, fc_absME_hr - np.sqrt(var_fc[hr_mask]), 'r')
    axes[3,1].plot(xc, an_absME_hr - np.sqrt(var_an[hr_mask]), 'b')
    axes[3,1].plot(xc, np.zeros(len(xc)),'k:')
    axes[3,1].set_xlabel('$x$',fontsize=18)

name_f = "/T%d_spr_err.png" %T
f_name_f = str(figsdir+name_f)
plt.savefig(f_name_f)
print(' ')
print(' *** %s at time level %d saved to %s' %(name_f,T,figsdir))


##################################################################
### 3 panel subplot: CRPS of 3 vars for fc and an
##################################################################

CRPS_fc = np.empty((Neq,Nk_fc))
CRPS_an = np.empty((Neq,Nk_fc))

for ii in h_mask:
    CRPS_fc[0,ii] = crps_calc(X[ii,:,T],X_tr[ii,0,T])
    CRPS_fc[1,ii] = crps_calc(X[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
    CRPS_fc[2,ii] = crps_calc(X[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])
    if(Neq==4): CRPS_fc[3,ii] = crps_calc(X[ii+3*Nk_fc,:,T],X_tr[ii+3*Nk_fc,0,T])
    CRPS_an[0,ii] = crps_calc(Xan[ii,:,T],X_tr[ii,0,T])
    CRPS_an[1,ii] = crps_calc(Xan[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
    CRPS_an[2,ii] = crps_calc(Xan[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])
    if(Neq==4): CRPS_an[3,ii] = crps_calc(Xan[ii+3*Nk_fc,:,T],X_tr[ii+3*Nk_fc,0,T])

lw = 1. # linewidth
axlim0 = np.max(CRPS_fc[0,:])
axlim1 = np.max(CRPS_fc[1,:])
axlim2 = np.max(CRPS_fc[2,:])
if(Neq==4): axlim3 = np.max(CRPS_fc[3,:])
 
ft = 16
xl = 0.65

fig, axes = plt.subplots(Neq, 1, figsize=(7,12))

axes[0].plot(xc, CRPS_fc[0,:],'r',linewidth=lw,label='fc')
axes[0].plot(xc, CRPS_an[0,:],'b',linewidth=lw,label='an')
axes[0].set_ylabel('$h(x)$',fontsize=18)
axes[0].text(xl, 1.2*axlim0, '$CRPS_{an} = %.3g$' %CRPS_an[0,:].mean(axis=-1), fontsize=ft, color='b')
axes[0].text(xl, 1.1*axlim0, '$CRPS_{fc} = %.3g$' %CRPS_fc[0,:].mean(axis=-1), fontsize=ft, color='r')
axes[0].set_ylim([0,1.3*axlim0])

axes[1].plot(xc, CRPS_fc[1,:],'r',linewidth=lw)
axes[1].plot(xc, CRPS_an[1,:],'b',linewidth=lw)
axes[1].set_ylabel('$u(x)$',fontsize=18)
axes[1].text(xl, 1.2*axlim1, '$CRPS_{an} = %.3g$' %CRPS_an[1,:].mean(axis=-1), fontsize=ft, color='b')
axes[1].text(xl, 1.1*axlim1, '$CRPS_{fc} = %.3g$' %CRPS_fc[1,:].mean(axis=-1), fontsize=ft, color='r')
axes[1].set_ylim([0,1.3*axlim1])

if(Neq==3):
    axes[2].plot(xc, CRPS_fc[2,:],'r',linewidth=lw)
    axes[2].plot(xc, CRPS_an[2,:],'b',linewidth=lw)
    axes[2].set_ylabel('$r(x)$',fontsize=18)
    axes[2].text(xl, 1.2*axlim2, '$CRPS_{an} = %.3g$' %CRPS_an[2,:].mean(axis=-1), fontsize=ft, color='b')
    axes[2].text(xl, 1.1*axlim2, '$CRPS_{fc} = %.3g$' %CRPS_fc[2,:].mean(axis=-1), fontsize=ft, color='r')
    axes[2].set_ylim([0,1.3*axlim2])
    axes[2].set_xlabel('$x$',fontsize=18)
if(Neq==4):
    axes[2].plot(xc, CRPS_fc[2,:],'r',linewidth=lw)
    axes[2].plot(xc, CRPS_an[2,:],'b',linewidth=lw)
    axes[2].set_ylabel('$v(x)$',fontsize=18)
    axes[2].text(xl, 1.2*axlim2, '$CRPS_{an} = %.3g$' %CRPS_an[2,:].mean(axis=-1), fontsize=ft, color='b')
    axes[2].text(xl, 1.1*axlim2, '$CRPS_{fc} = %.3g$' %CRPS_fc[2,:].mean(axis=-1), fontsize=ft, color='r')
    axes[2].set_ylim([0,1.3*axlim2])
    axes[2].set_xlabel('$x$',fontsize=18)
    axes[3].plot(xc, CRPS_fc[3,:],'r',linewidth=lw)
    axes[3].plot(xc, CRPS_an[3,:],'b',linewidth=lw)
    axes[3].set_ylabel('$r(x)$',fontsize=18)
    axes[3].text(xl, 1.2*axlim3, '$CRPS_{an} = %.3g$' %CRPS_an[3,:].mean(axis=-1), fontsize=ft, color='b')
    axes[3].text(xl, 1.1*axlim3, '$CRPS_{fc} = %.3g$' %CRPS_fc[3,:].mean(axis=-1), fontsize=ft, color='r')
    axes[3].set_ylim([0,1.3*axlim3])
    axes[3].set_xlabel('$x$',fontsize=18)

name = "/T%d_crps.png" %T
f_name = str(figsdir+name)
plt.savefig(f_name)
print(' ')
print(' *** %s at time level %d saved to %s' %(name,T,figsdir))
