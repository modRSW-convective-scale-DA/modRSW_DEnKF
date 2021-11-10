##################################################################
#--------------- Plotting routines for saved data ---------------
##################################################################
'''
    Plotting routine: <plot_func_t>
    
    Loads saved data in specific directories and produces plots as a function of time for OID, spr v err, and CRPS (i.e., domain-averaged time series). To use, specify (1) dir_name, (2) combination of parameters ijk.
    
    NOTE: Any changes to the outer loop parameters should be replicated here too.
    
    NOTE: currently saves as .png files

    CALL WITH "python plot_func_t.py <i> <j> <k> <l>

    Assumes only one RTPP value.
    '''

## generic modules 
import os
import errno
import importlib.util
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

## custom modules
#from parameters import *
from crps_calc_fun import crps_calc

##################################################################
##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

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
n_d = config.n_d
n_ens = config.n_ens
spin_up = config.spin_up

###################################################################
n_job = int(sys.argv[2])-1
print(n_job)
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
print(indices)
i = indices[n_job][0]
j = indices[n_job][1] 
k = indices[n_job][2]
l = indices[n_job][3]
###################################################################
lead_times=[3,4]

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
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles
Xforec = np.load(str(dirn+'/X_forec.npy')) # long term forecast
OI = np.load(str(dirn+'/OI.npy')) # OI

# LOAD TRUTH TRAJECTORY
U_tr = np.load(str(outdir+'/U_tr_array_2xres_1h.npy'))

# Manipulate U_tr to get X_tr
U_tr_tmp = np.empty((Neq,Nk_fc,Nforec+Nmeas))
for jj in range(0,Nk_fc):
    U_tr_tmp[:,jj,:] = U_tr[:,jj*dres,1:]
U_tr_tmp[1:,:,:] = U_tr_tmp[1:,:,:]/U_tr_tmp[0,:,:]

X_tr = np.empty((n_d,1,Nmeas+Nforec))
for kk in range(0,Nmeas+Nforec):
    X_tr[:,0,kk] = U_tr_tmp[:,:,kk].flatten()

# print shape of data arrays to terminal (sanity check)
print(' Check array shapes...')
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
#print('X_array shape (n_d,n_ens,T)      : ', np.shape(X))
print('X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr))
print('Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan))
print('X_forec array shape (n_d,n_ens,T,N_forec): ', np.shape(Xforec))
print(' ')
##################################################################

# determine parameters from loaded arrays
t_an = np.shape(Xan)[2]
time_vec = list(range(0,t_an))
print('time_vec = ', time_vec)
print(' ')

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
   

###################################################################
if(Neq==3): OI_check = 100*OI[1,1:]/3 + 100*OI[2,1:]/3 + 100*OI[3,1:]/3
if(Neq==4): OI_check = 100*OI[1,1:]/4 + 100*OI[2,1:]/4 + 100*OI[3,1:]/4 + 100*OI[4,1:]/4
 
OI_ave = 100*OI[0,1:-1].mean(axis=-1)

print('OI ave. =', OI_ave)

print(' ') 
print(' PLOT : OI')
print(' ') 
fig, axes = plt.subplots(1, 1, figsize=(8,5))
plt.suptitle("OI diagnostic  (N = %s): [loc, add_inf, rtps] = [%s, %s, %s]" % (n_ens,loc[i], add_inf[j], rtps[l]),fontsize=16)

axes.plot(time_vec[1:], 100*OI[1,1:],'r',label='$OID_h$') # rmse
axes.plot(time_vec[1:], 100*OI[2,1:],'b',label='$OID_{u}$')
if(Neq==3): 
    axes.plot(time_vec[1:], 100*OI[3,1:],'c',label='$OID_{r}$')
if(Neq==4):
    axes.plot(time_vec[1:], 100*OI[3,1:],'g',label='$OID_{v}$')
    axes.plot(time_vec[1:], 100*OI[4,1:],'c',label='$OID_{r}$')
axes.plot(time_vec[1:], 100*OI[0,1:],'k',linewidth=2.0,label='$OID$')
axes.set_ylabel('OID (%)',fontsize=18)
axes.legend(loc = 1, fontsize='large')
axes.set_ylim([0,100*np.max(OI[0,1:-1])])
axes.set_xlim([time_vec[1],time_vec[-1]])
axes.set_xlabel('Assim. time $T$',fontsize=14)

name = "/OID.png"
f_name = str(figsdir+name)
plt.savefig(f_name)
print(' ')
print(' *** %s saved to %s' %(name,figsdir))
print(' ')

##################################################################

# for means and deviations
Xanbar = np.empty(np.shape(Xan))
Xandev = np.empty(np.shape(Xan))
Xandev_tr = np.empty(np.shape(Xan))
Xforbar = np.empty(np.shape(Xforec))
Xfordev = np.empty(np.shape(Xforec))
Xfordev_tr = np.empty(np.shape(Xforec))

# for errs etc at each assim time

for_err = np.empty((n_d,len(lead_times)))
for_err2 = np.empty((n_d,len(lead_times)))
Pflong = np.empty((n_d,n_d,len(lead_times)))
Pflong_tr = np.empty((n_d,n_d,len(lead_times)))
var_longfc = np.empty((n_d,len(lead_times)))
var_longfct = np.empty((n_d,len(lead_times)))

rmse_fc = np.empty((Neq,len(time_vec),len(lead_times)))
rmse_an = np.empty((Neq,len(time_vec)))
spr_fc = np.empty((Neq,len(time_vec),len(lead_times)))
spr_an = np.empty((Neq,len(time_vec)))
ame_fc = np.empty((Neq,len(time_vec),len(lead_times)))
ame_an = np.empty((Neq,len(time_vec)))
tote_fc = np.empty((Neq,len(time_vec),len(lead_times)))
tote_an = np.empty((Neq,len(time_vec)))
crps_fc = np.empty((Neq,len(time_vec),len(lead_times)))
crps_an = np.empty((Neq,len(time_vec)))

# ausiliary crps array to compute means
CRPS_fc = np.empty((Neq,Nk_fc,len(lead_times)))
CRPS_an = np.empty((Neq,Nk_fc))


##################################################################
###               ERRORS + SPREAD                             ####
##################################################################

for T in time_vec:
    
    plt.clf() # clear figs from previous loop
    
#    Xanbar[:,:,T] = np.repeat(Xan[:,:,T].mean(axis=1), n_ens).reshape(n_d, n_ens)
#    Xandev[:,:,T] = Xan[:,:,T] - Xanbar[:,:,T] # an deviations from mean
#    Xandev_tr[:,:,T] = Xan[:,:,T] - X_tr[:,:,T] # an deviations from truth    
    
    # ANALYSIS: ensemble mean error
#    an_err = Xanbar[:,0,T] - X_tr[:,0,T] # an_err = analysis ens. mean - truth
#    an_err2 = an_err**2

    # analysis cov matrix
#    Pa = np.dot(Xandev[:,:,T],np.transpose(Xandev[:,:,T]))
#    Pa = Pa/(n_ens - 1) # analysis covariance matrix
#    var_an = np.diag(Pa)

#    Pa_tr = np.dot(Xandev_tr[:,:,T],np.transpose(Xandev_tr[:,:,T]))
#    Pa_tr = Pa_tr/(n_ens - 1) # fc covariance matrix w.r.t truth
#    var_ant = np.diag(Pa_tr)
   
    for n in range(len(lead_times)):
        ### CALCULATING ERRORS AT DIFFERENT LEAD TIMES ###
        Xforbar[:,:,T,lead_times[n]] = np.repeat(Xforec[:,:,T,lead_times[n]].mean(axis=1), n_ens).reshape(n_d, n_ens)
        Xfordev[:,:,T,lead_times[n]] = Xforec[:,:,T,lead_times[n]] - Xforbar[:,:,T,lead_times[n]]
        Xfordev_tr[:,:,T,lead_times[n]] = Xforec[:,:,T,lead_times[n]] - X_tr[:,:,T+lead_times[n]]

        # LONG-RANGE FORECAST: mean error
        for_err[:,n] = Xforbar[:,0,T,lead_times[n]] - X_tr[:,0,T+lead_times[n]]
        for_err2[:,n] = for_err[:,n]**2

        # LONG-RANGE FORECAST: cov matrix for spread...
        Pflong[:,:,n] = np.dot(Xfordev[:,:,T,lead_times[n]],np.transpose(Xfordev[:,:,T,lead_times[n]]))
        Pflong[:,:,n] = Pflong[:,:,n]/(n_ens - 1)
        var_longfc[:,n] = np.diag(Pflong[:,:,n])

        # ... and rmse
        Pflong_tr[:,:,n] = np.dot(Xfordev_tr[:,:,T,lead_times[n]],np.transpose(Xfordev_tr[:,:,T,lead_times[n]]))
        Pflong_tr[:,:,n] = Pflong_tr[:,:,n]/(n_ens - 1)
        var_longfct[:,n] = np.diag(Pflong_tr[:,:,n])

        ####                       CRPS                               ####
        ##################################################################
    
        for ii in h_mask:
            CRPS_fc[0,ii,n] = crps_calc(Xforec[ii,:,T,lead_times[n]],X_tr[ii,0,T+lead_times[n]])
            CRPS_fc[1,ii,n] = crps_calc(Xforec[ii+Nk_fc,:,T,lead_times[n]],X_tr[ii+Nk_fc,0,T+lead_times[n]])
            CRPS_fc[2,ii,n] = crps_calc(Xforec[ii+2*Nk_fc,:,T,lead_times[n]],X_tr[ii+2*Nk_fc,0,T+lead_times[n]])        
    
    #################################################################

        # domain-averaged stats (forecast)
        ame_fc[0,T,n] = np.mean(np.absolute(for_err[h_mask,n]))
        spr_fc[0,T,n] = np.sqrt(np.mean(var_longfc[h_mask,n]))
        rmse_fc[0,T,n] = np.sqrt(np.mean(for_err2[h_mask,n]))
        tote_fc[0,T,n] = np.sqrt(np.mean(var_longfct[h_mask,n]))
        crps_fc[0,T,n] = np.mean(CRPS_fc[0,:,n])
        
        ame_fc[1,T,n] = np.mean(np.absolute(for_err[hu_mask,n]))
        spr_fc[1,T,n] = np.sqrt(np.mean(var_longfc[hu_mask,n]))
        rmse_fc[1,T,n] = np.sqrt(np.mean(for_err2[hu_mask,n]))
        tote_fc[1,T,n] = np.sqrt(np.mean(var_longfct[hu_mask,n]))
        crps_fc[1,T,n] = np.mean(CRPS_fc[1,:,n])

        if(Neq==3):
            ame_fc[2,T,n] = np.mean(np.absolute(for_err[hr_mask,n]))
            spr_fc[2,T,n] = np.sqrt(np.mean(var_longfc[hr_mask,n]))
            rmse_fc[2,T,n] = np.sqrt(np.mean(for_err2[hr_mask,n]))
            tote_fc[2,T,n] = np.sqrt(np.mean(var_longfct[hr_mask,n]))
            crps_fc[2,T,n] = np.mean(CRPS_fc[2,:,n])

        if(Neq==4):
            ame_fc[2,T,n] = np.mean(np.absolute(for_err[hv_mask,n]))
            spr_fc[2,T,n] = np.sqrt(np.mean(var_longfc[hv_mask,n]))
            rmse_fc[2,T,n] = np.sqrt(np.mean(for_err2[hv_mask,n]))
            tote_fc[2,T,n] = np.sqrt(np.mean(var_longfct[hv_mask,n]))
            ame_fc[3,T,n] = np.mean(np.absolute(for_err[hr_mask,n]))
            spr_fc[3,T,n] = np.sqrt(np.mean(var_longfc[hr_mask,n]))
            rmse_fc[3,T,n] = np.sqrt(np.mean(for_err2[hr_mask,n]))
            tote_fc[3,T,n] = np.sqrt(np.mean(var_longfct[hr_mask,n]))
            crps_fc[2,T,n] = np.mean(CRPS_fc[2,:,n])
            crps_fc[3,T,n] = np.mean(CRPS_fc[3,:,n])

        
    ####                       CRPS                               ####
    ##################################################################

#    for ii in h_mask:
#        CRPS_an[0,ii] = crps_calc(Xan[ii,:,T],X_tr[ii,0,T])
#        CRPS_an[1,ii] = crps_calc(Xan[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
#        CRPS_an[2,ii] = crps_calc(Xan[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])

    # domain-averaged stats (analysis)
#    ame_an[0,T] = np.mean(np.absolute(an_err[h_mask]))
#    spr_an[0,T] = np.sqrt(np.mean(var_an[h_mask]))
#    rmse_an[0,T] = np.sqrt(np.mean(an_err2[h_mask]))
#    tote_an[0,T] = np.sqrt(np.mean(var_ant[h_mask]))
#    crps_an[0,T] = np.mean(CRPS_an[0,:])

#    ame_an[1,T] = np.mean(np.absolute(an_err[hu_mask]))
#    spr_an[1,T] = np.sqrt(np.mean(var_an[hu_mask]))
#    rmse_an[1,T] = np.sqrt(np.mean(an_err2[hu_mask]))
#    tote_an[1,T] = np.sqrt(np.mean(var_ant[hu_mask]))
#    crps_an[1,T] = np.mean(CRPS_an[1,:])

#    ame_an[2,T] = np.mean(np.absolute(an_err[hr_mask]))
#    spr_an[2,T] = np.sqrt(np.mean(var_an[hr_mask]))
#    rmse_an[2,T] = np.sqrt(np.mean(an_err2[hr_mask]))
#    tote_an[2,T] = np.sqrt(np.mean(var_ant[hr_mask]))
#    crps_an[2,T] = np.mean(CRPS_an[2,:])
#####################################################################

###########################################################################

col_vec = cm.rainbow(np.linspace(0,1,len(lead_times)))

print(' ')
print(' PLOT : RMS ERRORS vs SPREAD')
print(' ')
ft=16

axlim0 = 0.2#np.max(np.maximum(spr_fc[0,:-lead_times[-1],len(lead_times)-1], rmse_fc[0,:-lead_times[-1],len(lead_times)-1]))
axlim1 = 0.08#np.max(np.maximum(spr_fc[1,:-lead_times[-1],len(lead_times)-1], rmse_fc[1,:-lead_times[-1],len(lead_times)-1]))
axlim2 = 0.007#np.max(np.maximum(spr_fc[2,:-lead_times[-1],len(lead_times)-1], rmse_fc[2,:-lead_times[-1],len(lead_times)-1]))
if(Neq==4): axlim3 = np.max(np.maximum(spr_fc[3,:-lead_times[-1],len(lead_times)-1], rmse_fc[3,:-lead_times[-1],len(lead_times)-1]))

time_vec = list(range(0,t_an+3))

fig, axes = plt.subplots(Neq, 1, figsize=(7,12))
plt.suptitle("Domain-averaged error vs spread  (N = %s): \n [loc, add_inf, rtps] = [%s, %s, %s]" % (n_ens,loc[i], add_inf[j], rtps[l]),fontsize=16)

for ii in range(len(lead_times)):
    axes[0].plot(time_vec[lead_times[ii]:], spr_fc[0, :len(time_vec)-lead_times[ii], ii], color=col_vec[ii], label=str('fc+'+str(lead_times[ii])+' spread'))
    axes[0].set_ylabel('$h$',fontsize=18)
    axes[0].text(len(lead_times)+2, (1.1+ii/10.)*axlim0, '$(SPR,RMSE)_{fc+'+str(lead_times[ii])+'} = (%.3g,%.3g)$' %(spr_fc[0,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1),rmse_fc[0,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1)), fontsize=ft, color=col_vec[ii])
    axes[0].set_ylim([0,1.3*axlim0])
    #axes[0].legend(loc = 4, fontsize='small')
    axes[1].plot(time_vec[lead_times[ii]:], spr_fc[1,:len(time_vec)-lead_times[ii], ii], color=col_vec[ii], label=str('fc+'+str(lead_times[ii])+' spread'))
    axes[1].set_ylabel('$u$',fontsize=18)
    axes[1].text(len(lead_times)+2, (1.1+ii/10.)*axlim1, '$(SPR,RMSE)_{fc+'+str(lead_times[ii])+'} = (%.3g,%.3g)$' %(spr_fc[1,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1),rmse_fc[1,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1)), fontsize=ft, color=col_vec[ii])
    axes[1].set_ylim([0,1.3*axlim1])
    if(Neq==3):
        axes[2].plot(time_vec[lead_times[ii]:], spr_fc[2,:len(time_vec)-lead_times[ii], ii], color=col_vec[ii], label=str('fc+'+str(lead_times[ii])+' spread'))
        axes[2].set_ylabel('$r$',fontsize=18)
        axes[2].set_xlabel('Assim. time $T$',fontsize=14)
        axes[2].text(len(lead_times)+2, (1.1+ii/10.)*axlim2, '$(SPR,RMSE)_{fc+'+str(lead_times[ii])+'} = (%.3g,%.3g)$' %(spr_fc[2,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1),rmse_fc[2,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1)), fontsize=ft, color=col_vec[ii])
        axes[2].set_ylim([0,1.3*axlim2])
    if(Neq==4):
        axes[2].plot(time_vec[lead_times[ii]:], spr_fc[2,:len(time_vec)-lead_times[ii], ii], color=col_vec[ii], label=str('fc+'+str(lead_times[ii])+' spread'))
        axes[2].set_ylabel('$v$',fontsize=18)
        axes[2].text(len(lead_times)+2, (1.1+ii/10.)*axlim2, '$(SPR,RMSE)_{fc+'+str(lead_times[ii])+'} = (%.3g,%.3g)$' %(spr_fc[2,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1),rmse_fc[2,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1)), fontsize=ft, color=col_vec[ii])
        axes[2].set_ylim([0,1.3*axlim2])
        axes[3].plot(time_vec[lead_times[ii]:], spr_fc[3,:len(time_vec)-lead_times[ii], ii], color=col_vec[ii], label=str('fc+'+str(lead_times[ii])+' spread'))
        axes[3].set_ylabel('$r$',fontsize=18)
        axes[3].set_xlabel('Assim. time $T$',fontsize=14)
        axes[3].text(len(lead_times)+2, (1.1+ii/10.)*axlim3, '$(SPR,RMSE)_{fc+'+str(lead_times[ii])+'} = (%.3g,%.3g)$' %(spr_fc[3,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1),rmse_fc[3,spin_up:len(time_vec)-lead_times[ii],ii].mean(axis=-1)), fontsize=ft, color=col_vec[ii])
        axes[3].set_ylim([0,1.3*axlim3])


for jj in range(len(lead_times)):
    axes[0].plot(time_vec[lead_times[jj]:], rmse_fc[0, :len(time_vec)-lead_times[jj], jj], color=col_vec[jj], linestyle='dashed', label=str('fc+'+str(lead_times[ii])+' rmse'))
    axes[1].plot(time_vec[lead_times[jj]:], rmse_fc[1, :len(time_vec)-lead_times[jj], jj], color=col_vec[jj], linestyle='dashed', label=str('fc+'+str(lead_times[ii])+' rmse'))
    if(Neq==3):
        axes[2].plot(time_vec[lead_times[jj]:], rmse_fc[2, :len(time_vec)-lead_times[jj], jj], color=col_vec[jj], linestyle='dashed', label=str('fc+'+str(lead_times[ii])+' rmse'))
    if(Neq==4):
        axes[2].plot(time_vec[lead_times[jj]:], rmse_fc[2, :len(time_vec)-lead_times[jj], jj], color=col_vec[jj], linestyle='dashed', label=str('fc+'+str(lead_times[ii])+' rmse'))
        axes[3].plot(time_vec[lead_times[jj]:], rmse_fc[3, :len(time_vec)-lead_times[jj], jj], color=col_vec[jj], linestyle='dashed', label=str('fc+'+str(lead_times[ii])+' rmse'))

#axes[0].plot(time_vec, rmse_an[0,:], color='black', linestyle='dashed', label=str('an rmse'))
#axes[0].plot(time_vec, spr_an[0,:], color='black', label=str('an spread'))
#axes[1].plot(time_vec, rmse_an[1,:], color='black', linestyle='dashed', label=str('an rmse'))
#axes[1].plot(time_vec, spr_an[1,:], color='black', label=str('an spread'))
#axes[2].plot(time_vec, rmse_an[2,:], color='black', linestyle='dashed', label=str('an rmse'))
#axes[2].plot(time_vec, spr_an[2,:], color='black', label=str('an spread'))

name = "/spr_err.png"
f_name = str(figsdir+name)
plt.savefig(f_name)
print(' ')
print(' *** %s saved to %s' %(name,figsdir))
print(' ')

exit()
###########################################################################
###########################################################################

print(' ')
print(' PLOT : CRPS')
print(' ')

axlim0 = np.max(crps_fc[0,:-7,4])
axlim1 = np.max(crps_fc[1,:-7,4])
axlim2 = np.max(crps_fc[2,:-7,4])
if(Neq==4): axlim3 = np.max(crps_fc[3,:-7,4])

fig, axes = plt.subplots(Neq, 1, figsize=(7,12))
plt.suptitle("Domain-averaged CRPS  (N = %s): \n [loc, add_inf, rtps] = [%s, %s, %s]" % (n_ens,loc[i], add_inf[j], rtps[l]),fontsize=16)

axes[0].plot(time_vec, crps_fc[0,:,0],'r',label='fc+0') # spread
axes[0].plot(time_vec, crps_an[0,:],'b',label='an')
axes[0].set_ylabel('$h$',fontsize=18)
axes[0].text(1, 1.2*axlim0, '$CRPS_{an} = %.3g$' %crps_an[0,:].mean(axis=-1), fontsize=ft, color='b')
axes[0].text(1, 1.1*axlim0, '$CRPS_{fc+0} = %.3g$' %crps_fc[0,:,0].mean(axis=-1), fontsize=ft, color='r')
axes[0].set_ylim([0,1.3*axlim0])

axes[1].plot(time_vec, crps_fc[1,:,0],'r',label='fc+0') # spread
axes[1].plot(time_vec, crps_an[1,:],'b',label='an')
axes[1].set_ylabel('$u$',fontsize=18)
axes[1].text(1, 1.2*axlim1, '$CRPS_{an} = %.3g$' %crps_an[1,:].mean(axis=-1), fontsize=ft, color='b')
axes[1].text(1, 1.1*axlim1, '$CRPS_{fc+0} = %.3g$' %crps_fc[1,:,0].mean(axis=-1), fontsize=ft, color='r')
axes[1].set_ylim([0,1.3*axlim1])

if(Neq==3):
    axes[2].plot(time_vec, crps_fc[2,:,0],'r',label='fc+0') # spread
    axes[2].plot(time_vec, crps_an[2,:],'b',label='an')
    axes[2].set_ylabel('$r$',fontsize=18)
    axes[2].set_xlabel('Assim. time $T$',fontsize=14)
    axes[2].text(1, 1.2*axlim2, '$CRPS_{an} = %.3g$' %crps_an[2,:].mean(axis=-1), fontsize=ft, color='b')
    axes[2].text(1, 1.1*axlim2, '$CRPS_{fc+0} = %.3g$' %crps_fc[2,:,0].mean(axis=-1), fontsize=ft, color='r')
    axes[2].set_ylim([0,1.3*axlim2])
if(Neq==4):
    axes[2].plot(time_vec, crps_fc[2,:,0],'r',label='fc+0') # spread
    axes[2].plot(time_vec, crps_an[2,:],'b',label='an')
    axes[2].set_ylabel('$v$',fontsize=18)
    axes[2].set_xlabel('Assim. time $T$',fontsize=14)
    axes[2].text(1, 1.2*axlim2, '$CRPS_{an} = %.3g$' %crps_an[2,:].mean(axis=-1), fontsize=ft, color='b')
    axes[2].text(1, 1.1*axlim2, '$CRPS_{fc+0} = %.3g$' %crps_fc[2,:,0].mean(axis=-1), fontsize=ft, color='r')
    axes[2].set_ylim([0,1.3*axlim2])
    axes[3].plot(time_vec, crps_fc[3,:,0],'r',label='fc+0') # spread
    axes[3].plot(time_vec, crps_an[3,:],'b',label='an')
    axes[3].set_ylabel('$r$',fontsize=18)
    axes[3].set_xlabel('Assim. time $T$',fontsize=14)
    axes[3].text(1, 1.2*axlim3, '$CRPS_{an} = %.3g$' %crps_an[3,:].mean(axis=-1), fontsize=ft, color='b')
    axes[3].text(1, 1.1*axlim3, '$CRPS_{fc+0} = %.3g$' %crps_fc[3,:,0].mean(axis=-1), fontsize=ft, color='r')
    axes[3].set_ylim([0,1.3*axlim2])

name = "/crps.png"
f_name = str(figsdir+name)
plt.savefig(f_name)
print(' ')
print(' *** %s saved to %s' %(name,figsdir))
print(' ')

###########################################################################
