##################################################################
#--------- Plot err doub time histogram from EFS stats -----------
##################################################################
'''
    Plots error doubling time histograms from saved data <err_doub_Tn.npy>
    
    '''


## generic modules
import os
import errno
import importlib.util
import itertools
import sys
import numpy as np
import matplotlib.pyplot as plt

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
n_ens = config.n_ens

###################################################################
# Read experiment index from command line and split it into indeces
###################################################################

n_job = int(sys.argv[2])-1
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
i = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

##################################################################
# make fig directory (if it doesn't already exist)
dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]),
                                                 str(add_inf[j]), str(rtpp[k]),
                                                 str(rtps[l]))
dirEDT = str(dirn+'/EDT')
figsdir = str(dirEDT+'/figs')

#check if dir exixts, if not make it
try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

T0 = list(range(25))
err_doub = np.empty((Neq,len(T0)*n_ens))
#
for i in range(0,len(T0)):
    err_doub[:,i*n_ens:(i+1)*n_ens] = np.load(str(dirEDT+'/err_doub_T'+str(T0[i]))+'.npy')
#

###############################################################################
print(' ')
print(' PLOT : Error-doubling time histograms...')
print(' ')

# bin width = 1hr
print('... with bin width = 1hr ')

print(err_doub[0,~np.isnan(err_doub[0,:])])

var = ['h','u','r']

fig, axes = plt.subplots(3, 1, figsize=(7,12))
plt.suptitle('Error-doubling time: histogram',fontsize=18)
for kk in range(0,Neq):
    hist, bins = np.histogram(err_doub[kk,~np.isnan(err_doub[kk,:])], bins = np.linspace(1,25,25))
    axes[kk].hist(err_doub[kk,~np.isnan(err_doub[kk,:])], bins = np.linspace(1,24,24))
    axes[kk].set_xlim([1,25])
    axes[kk].tick_params(axis='x',labelsize=16)
    axes[kk].tick_params(axis='y',labelsize=16)
    axes[kk].set_ylabel('Count ('+var[kk]+')',fontsize=16)
    axes[kk].text(16, 0.8*np.max(hist), '$Mean = %.3g$' %np.nanmean(err_doub[kk,~np.isnan(err_doub[kk,:])]), fontsize=16, color='b')
    axes[kk].text(16, 0.7*np.max(hist), '$Median = %.3g$' %np.median(err_doub[kk,~np.isnan(err_doub[kk,:])]), fontsize=16, color='b')
    axes[kk].text(16, 0.9*np.max(hist), '$Total = %.3g / %.3g$' %(np.sum(hist), len(T0)*n_ens), fontsize=16, color='b')
axes[2].set_xlabel('Time (hrs)',fontsize=16)

name = '/err_doub_hist_ave.pdf'
f_name = str(figsdir+name)
plt.savefig(f_name)
print(' ')
print(' *** %s saved to %s' %(name,figsdir))
print(' ')


# bin width = 2hr
print('... with bin width = 2hr ')

fig, axes = plt.subplots(3, 1, figsize=(7,12))
plt.suptitle('Error-doubling time: histogram',fontsize=18)
for kk in range(0,Neq):
    hist, bins = np.histogram(err_doub[kk,~np.isnan(err_doub[kk,:])], bins = np.linspace(1,25,13))
    axes[kk].hist(err_doub[kk,~np.isnan(err_doub[kk,:])], bins = np.linspace(1,25,13))
    axes[kk].set_xlim([1,25])
    axes[kk].tick_params(axis='x',labelsize=16)
    axes[kk].tick_params(axis='y',labelsize=16)
    axes[kk].set_ylabel('Count ('+var[kk]+')',fontsize=16)
    axes[kk].text(16, 0.8*np.max(hist), '$Mean = %.3g$' %np.nanmean(err_doub[kk,~np.isnan(err_doub[kk,:])]), fontsize=16, color='b')
    axes[kk].text(16, 0.7*np.max(hist), '$Median = %.3g$' %np.median(err_doub[kk,~np.isnan(err_doub[kk,:])]), fontsize=16, color='b')
    axes[kk].text(16, 0.9*np.max(hist), '$Total = %.3g / %.3g$' %(np.sum(hist), len(T0)*n_ens), fontsize=16, color='b')
axes[2].set_xlabel('Time (hrs)',fontsize=16)

name = '/err_doub_hist_ave2.pdf'
f_name = str(figsdir+name)
plt.savefig(f_name)
print(' ')
print(' *** %s saved to %s' %(name,figsdir))
print(' ')

#np.isnan(X[0,:]).sum()
#print np.sum(hist)
###########################################################################
