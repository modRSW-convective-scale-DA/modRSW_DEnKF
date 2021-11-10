#######################################################################
# modRSW model with topography compatible with python3
#######################################################################
'''
Given mesh, IC, time paramters, integrates modRSW and plots evolution. 
Useful first check of simulations before use in the EnKF.
'''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################

import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy as sp
import importlib.util
import matplotlib.pyplot as plt
import errno 
import sys
import os

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

from f_modRSW import step_forward_topog, time_step, make_grid, make_grid_2, step_forward_modRSW, step_forward_modRSW_inflow
from init_cond_modRSW import init_cond_topog_cos, init_cond_topog, init_cond_4, init_cond_rest

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

Neq = config.Neq
Nk = config.Nk
ic = config.ic
outdir = config.outdir
cfl_fc = config.cfl
cc2 = config.cc2
beta = config.beta
alpha2 = config.alpha2
g = config.g
H0 = config.H0
#H_pert = config.H_pert
L = config.L
A = config.A
V = config.V
tn = config.tn
#dt = config.dt
#kx1 = config.kx1
Ro = config.Ro
Fr = config.Fr
Hc = config.Hc
Hr = config.Hr
Nmeas = config.Nmeas
tmax = config.tmax
dtmeasure = config.dtmeasure
tmeasure = config.tmeasure
U_relax = config.U_relax
tau_rel = config.tau_rel

print(' -------------- ------------- ------------- ------------- ')
print(' --- TEST CASE: model only (dynamics and integration) --- ')
print(' -------------- ------------- ------------- ------------- ')
print(' ')
print(' Number of elements Nk =', Nk)
print(' Initial condition:', ic)
print(' ')

#################################################################
# create directory for output
#################################################################

dirname = str('/test_model')
dirn = str(outdir+dirname)
#check if dir exixts, if not make it
try:
    os.makedirs(dirn)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

##################################################################    
# Mesh generation for forecast and truth resolutions
##################################################################
grid =  make_grid(Nk,L) # forecast
Kk = grid[0]
x = grid[1]
xc = grid[2]

##################################################################    
#'''%%%----- Apply initial conditions -----%%%'''
##################################################################
print('Generate initial conditions...')
U0,B = ic(x,Nk,Neq,H0,L,A,V)

# Frequency of inertia-gravity wave used as an inflow on one end
#w1 = np.sqrt((1./Ro)**2+(H0/Fr**2)*kx1**2)

# 
#U0[0,:] = H0+H_pert*np.cos(kx1*xc)
#U0[1,:] = H_pert*(w1/(H0*kx1))*np.cos(kx1*xc)*H0
#U0[2,:] = H_pert*(1./(Ro*H0*kx1))*np.sin(kx1*xc)*H0

### 4 panel subplot for initial state of 4 vars
fig, axes = plt.subplots(4, 1, figsize=(8,8))
plt.suptitle("Initial condition with Nk = %d" %Nk)

axes[0].plot(xc, U0[0,:]+B, 'b')
#axes[0].plot(xc, B, 'k', linewidth=2.)
axes[0].set_ylabel('$h_0(x)$',fontsize=18)
#axes[0].set_ylim([0,2*H0])

axes[1].plot(xc, U0[1,:]/U0[0,:], 'b')
#axes[1].set_ylim([-2,2])
axes[1].set_ylabel('$u_0(x)$',fontsize=18)

#axes[2].plot(xc, U0[3,:]/U0[0,:], 'b')
#axes[2].set_ylim([-2,2])
#axes[2].set_ylabel('$v_0(x)$',fontsize=18)

if(Neq==3):
    axes[2].plot(xc, U0[2,:]/U0[0,:], 'b')
    axes[2].set_ylabel('$r_0(x)$',fontsize=18)
    #axes[2].set_ylim([-0.025,0.15])
    axes[2].set_xlabel('$x$',fontsize=18)

if(Neq==4):
    axes[2].plot(xc, U0[2,:]/U0[0,:], 'b')
    axes[2].set_ylabel('$v_0(x)$',fontsize=18)
    #axes[2].set_ylim([-0.025,0.15])
    axes[2].set_xlabel('$x$',fontsize=18)

    axes[3].plot(xc, U0[3,:]/U0[0,:], 'b')
    axes[3].set_ylabel('$r_0(x)$',fontsize=18)
    axes[3].set_xlabel('$x$',fontsize=18)

#plt.show() # use block=False?

name_fig = "/ic_Nk%d.png" %Nk
f_name_fig = str(dirn+name_fig)
plt.savefig(f_name_fig)
print(' *** Initial condition %s saved to %s' %(name_fig,dirn))
print(' ')

##################################################################
#'''%%%----- Define system arrays and time parameters------%%%'''
##################################################################
U = np.empty([Neq,Nk])
U = U0
index = 1

### Relaxation solution ###
U_rel = U_relax(Neq,Nk,L,V,xc,U)

##################################################################
#'''%%%----- integrate forward in time until tmax ------%%%'''
##################################################################
U_array = np.empty((Neq,Nk,Nmeas+1))
U_array[:,:,0] = U0
U_hovplt = np.zeros([Neq,Nk,120000])
t_hovplt = np.zeros(120000)
p=0
i=0
#dt = 5.28e-05

print(' ')
print('Integrating forward from t =', tn, 'to', tmeasure,'...')
while tn < tmax:

 #   dt_old = np.copy(dt)

    dt = time_step(U,Kk,cfl_fc,cc2,beta,g)
    tn = tn + dt

   # dtmin = min(dt,dt_old)

    print(tn)
#    U_infl = [H0+H_pert*np.cos(kx1*0.-w1*tn),(w1/(H0*kx1))*H_pert*np.cos(kx1*0.-w1*tn),(1./(Ro*H0*kx1))*H_pert*np.sin(kx1*0.-w1*tn)]
#    U_infl[1] = U_infl[0]*U_infl[1]
#    U_infl[2] = U_infl[0]*U_infl[2]

#    if tn > tmeasure:
#        dt = dt - (tn - tmeasure) + 1e-12
#        tn = tmeasure + 1e-12

    U = step_forward_topog(U,B,dt,tn,Neq,Nk,Kk,Hc,Hr,cc2,beta,alpha2,g)
#    U = step_forward_modRSW_inflow(U,dt,Neq,Nk,Kk,Ro,alpha2,Hc,Hr,cc2,beta,g,U_infl)
#    U = step_forward_modRSW(U,U_rel,dt,Neq,Nk,Kk,Ro,alpha2,Hc,Hr,cc2,beta,g,tau_rel)

    # Save data for hovmoller plot
    if(p%3==0 and i<120000):
        U_hovplt[:,:,i] = U
        t_hovplt[i] = tn-dt
        i = i+1
    p = p+1

    if tn > tmeasure:
        print(' ')
        print('Plotting at time =',tmeasure)

        fig, axes = plt.subplots(4, 1, figsize=(6,8))
        plt.suptitle("Model trajectories at t = %.3f with Nk =%d" %(tmeasure,Nk))
        
        axes[0].plot(xc, U[0,:]+B, 'b')
#        axes[0].plot(xc, B, 'k', linewidth=2.0)
#        axes[0].plot(xc,Hc*np.ones(len(xc)),'r:')
#        axes[0].plot(xc,Hr*np.ones(len(xc)),'r:')
#        axes[0].set_ylim([0,3*H0])
        axes[0].set_ylabel('$h(x)$',fontsize=18)
        
        axes[1].plot(xc, U[1,:]/U[0,:], 'b')
#        axes[1].set_ylim([-2,2])
        axes[1].set_ylabel('$u(x)$',fontsize=18)

        if(Neq==3):
            axes[2].plot(xc, U[2,:]/U[0,:], 'b')
            axes[2].plot(xc,np.zeros(len(xc)),'k--')
            axes[2].set_ylabel('$r(x)$',fontsize=18)
#           axes[2].set_ylim([-0.025,0.15])
            axes[2].set_xlabel('$x$',fontsize=18)

        if(Neq==4):
            axes[2].plot(xc, U[2,:]/U[0,:], 'b')
            axes[2].plot(xc,np.zeros(len(xc)),'k--')
            axes[2].set_ylabel('$v(x)$',fontsize=18)
#           axes[2].set_ylim([-0.025,0.15])
            axes[2].set_xlabel('$x$',fontsize=18)
    
            axes[3].plot(xc, U[3,:]/U[0,:], 'b')
            #axes[3].set_ylim([-2,2])
            axes[3].set_ylabel('$r(x)$',fontsize=18)

        name_fig = "/t%d_Nk%d.png" %(index, Nk)
        f_name_fig = str(dirn+name_fig)
        plt.savefig(f_name_fig)
        print(' *** %s at time level %d saved to %s' %(name_fig,index,dirn))
        
        U_array[:,:,index] = U
        
        index = index + 1
        tmeasure = tmeasure + dtmeasure
        print(' ')
        print('Integrating forward from t =', tmeasure-dtmeasure, 'to', tmeasure,'...')

print(' ')
print('***** DONE: end of simulation at time:', tn)
print(' ')

print(' Saving simulation data in:', dirn)

np.save(str(dirn+'/U_array_Nk%d' %Nk),U_array)
np.save(str(dirn+'/B_Nk%d' %Nk),B)
np.save(str(outdir+'/U_hovplt'),U_hovplt)
np.save(str(outdir+'/t_hovplt'),t_hovplt)

print(' ')
print(' CHECK data value: maximum h(x) at t = 0.288:' , np.max(U_array[0,:,2]), ' at x = ', xc[np.argmax(U_array[0,:,2])])
print(' ')
print(' -------------- SUMMARY: ------------- ')  
print(' ') 
print('Dynamics:')
print('Ro =', Ro)  
print('Fr = ', Fr)
print('(H_0 , H_c , H_r) =', [H0, Hc, Hr])  
print(' Mesh: number of elements Nk =', Nk)
#print(' Minimum time step =', dtmin)
print(' ')   
print(' ----------- END OF SUMMARY ---------- ')
print(' ')  


##################################################################
#			END OF PROGRAM				 #
##################################################################

