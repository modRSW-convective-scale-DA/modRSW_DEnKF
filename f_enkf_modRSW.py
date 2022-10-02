#######################################################################
# Collection of functions used in EnKF scripts
#######################################################################
'''
Functions:
> generate_truth(): generates nature run at refined resolution (Nk_tr) and saves run at given observing times (assim_time)
> gasp_cohn(): gaspari cohn taper function
>  analysis_step_enkf(): updates ensemble given observations from truth, returns analysis ensemble and forecast ensemble.
'''
import math as m
import numpy as np
#from parameters import *
import os

##################################################################
#------------------ Compute nature run ------------------
##################################################################


def generate_truth(U_tr_array, U_rel, Neq, Nk_tr, tr_grid, cfl_tr, assim_time, tmax, f_path_name, Hc, Hr, cc2, beta, alpha2, g, Ro, tau_rel):
#def generate_truth(U_tr_array, B_tr, Neq, Nk_tr, tr_grid, cfl_tr, assim_time, tmax, f_path_name, Hc, Hr, cc2, beta, alpha2, g):
    
    from f_modRSW import time_step, step_forward_topog, step_forward_modRSW

    Kk_tr = tr_grid[0] 
    x_tr = tr_grid[1]
    
    tn = 0.0
    #tmax = assim_time[-1]
    dtmeasure = assim_time[1]-assim_time[0]
    tmeasure = dtmeasure
    
    U_tr = U_tr_array[:,:,0]
    
    print(' ')
    print('Integrating forward from t =', tn, 'to', tmax,'...')
    print(' ')
    
    index = 1 # for U_tr_array (start from 1 as 0 contains IC).
    while tn < tmax:
        dt = time_step(U_tr,Kk_tr,cfl_tr,cc2,beta,g)
        tn = tn + dt

        if tn > tmeasure:
            dt = dt - (tn - tmeasure) + 1e-12
            tn = tmeasure + 1e-12

        U_tr = step_forward_modRSW(U_tr,U_rel,dt,Neq,Nk_tr,Kk_tr,Ro,alpha2,Hc,Hr,cc2,beta,g,tau_rel)
#        U_tr = step_forward_topog(U_tr,B_tr,dt,tn,Neq,Nk_tr,Kk_tr,Hc,Hr,cc2,beta,alpha2,g)
#        print 't_tr =',tn

        if tn > tmeasure:
            U_tr_array[:,:,index] = U_tr
            print('*** STORE TRUTH at observing time = ',tmeasure,' ***')
            tmeasure = tmeasure + dtmeasure
            index = index + 1
            
    np.save(f_path_name,U_tr_array)
    
    print('* DONE: truth array saved to:', f_path_name, ' with shape:', np.shape(U_tr_array), ' *')
        
    return U_tr_array    
    
    
##################################################################
# GASPARI-COHN TAPER FUNCTION FOR COV LOCALISATION
# from Jeff Whitaker's github: https://github.com/jswhit/
################################################################    
def gaspcohn(r):
    # Gaspari-Cohn taper function
    # very close to exp(-(r/c)**2), where c = sqrt(0.15)
    # r should be >0 and normalized so taper = 0 at r = 1
    rr = 2.0*r
    rr += 1.e-13 # avoid divide by zero warnings from numpy
    taper = np.where(r<=0.5, \
                     ( ( ( -0.25*rr +0.5 )*rr +0.625 )*rr -5.0/3.0 )*rr**2 + 1.0,\
                     np.zeros(r.shape,r.dtype))

    taper = np.where(np.logical_and(r>0.5,r<1.), \
                    ( ( ( ( rr/12.0 -0.5 )*rr +0.625 )*rr +5.0/3.0 )*rr -5.0 )*rr \
                    + 4.0 - 2.0 / (3.0 * rr), taper)
    return taper    

##################################################################
# GASPARI-COHN matrix using taper function
################################################################ 
def gaspcohn_matrix(loc_rho,Nk_fc,Neq):
    # construct localisation matrix rho based on Gaspari Cohn function
    
    rr = np.arange(0,loc_rho,loc_rho/Nk_fc) 
    vec = gaspcohn(rr)
    
    rho = np.zeros((Nk_fc,Nk_fc))
    
    for i in range(Nk_fc):
        for j in range(Nk_fc):
            rho[i,j] = vec[np.min([abs(i-j),abs(i+Nk_fc-j),abs(i-Nk_fc-j)])]
    
    rho = np.tile(rho, (Neq,Neq))
    
    return rho    

##################################################################
#'''------------------ ANALYSIS STEP ------------------'''
##################################################################

def analysis_step_enkf_v3(U_fc, U_tr, Y_obs, H, tmeasure, dtmeasure, index, pars_enda, pars):
    '''
        (Steps refer to algorithm on page 121 of thesis, as do eq. numbers)
        
        INPUTS
        U_fc: ensemble trajectories in U space, shape (Neq,Nk_fc,n_ens)
        U_tr: truth trajectory in U space, shape (Neq,Nk_tr,Nmeas+1)
        tmeasure: time of assimilation
        dtmeasure: length of window
        pars_enda: vector of parameters relating to DA fixes (relaxation and localisation)
        '''
    
    print(' ')
    print('----------------------------------------------')
    print('------------ ANALYSIS STEP: START ------------')
    print('----------------------------------------------')
    print(' ')

    L = pars[14]
    Nk_fc = pars[0]
    n_ens = pars[2]
    Neq = pars[13]
    dres = pars[8]
    n_obs = pars[6]
    n_obs_h = pars[9]
    n_obs_u = pars[10]
    n_obs_r = pars[11]
    n_d = pars[12]  
    ob_noise = pars[15]
    h_obs_mask = pars[16]
    hu_obs_mask = pars[17]
    hr_obs_mask = pars[18]

    #Nk_fc = np.shape(U_fc)[1] # fc resolution (no. of cells)
    Kk_fc = L/Nk_fc # fc resolution (cell length)
    rtpp = pars_enda[0] # relaxation factor
    rtps = pars_enda[1] # relaxation factor
    loc = pars_enda[2]
    add_inf = pars_enda[3]
    
    print(' ')
    print('--------- ANALYSIS: EnKF ---------')
    print(' ')
    print('Assimilation time = ', tmeasure)
    print('Number of ensembles = ', n_ens)
    
    # project truth onto forecast grid so that U and U_tr are the same dimension
    U_tmp = np.empty([Neq,Nk_fc])
    for i in range(0,Nk_fc):
        U_tmp[:,i] = U_tr[:, i*dres, index+1]
    U_tr = U_tmp

    '''
        step 1.(c)
        '''
    # for assimilation, work with [h,u,r]
    U_fc[1:,:,:] = U_fc[1:,:,:]/U_fc[0,:,:]
    U_tr[1:,:] = U_tr[1:,:]/U_tr[0,:]
    
    X_tr = U_tr.flatten()
    X_tr = X_tr.T
    
    # state matrix (flatten the array)
    X = np.empty([n_d,n_ens])
    for N in range(0,n_ens):
        X[:,N] = U_fc[:,:,N].flatten()
    

    '''
        Step 2.(a)
        '''
    # Add observation perturbations to each member, ensuring that the
    # observation perturbations have zero mean over all members to
    # avoid perturbing the ensemble mean. Do not apply when rtpp factor is 0.5
    # as Sakov and Oke (2008) results are equivalent to saying that rtpp 0.5
    # gives a deterministic ensemble Kalman filter in which perturbation
    # observations should not be applied.
    #print(ob_noise)
    #ob_noise = np.repeat(ob_noise, [n_obs_h,n_obs_u,n_obs_r])
    if rtpp != 0.5:
        obs_pert = np.repeat(ob_noise, [n_obs_h,n_obs_u,n_obs_r])[:, None] * np.random.randn(n_obs, n_ens)
        obs_pert_mean = np.mean(obs_pert, axis=1)
        obs_pert -= np.repeat(obs_pert_mean, n_ens).reshape(n_obs, n_ens)

        print('obs_pert shape =', np.shape(obs_pert))
    
        # y_o = y_m + e_o (eq. 6.6), with y_m itself a perturbed observation.
        # N.B. The observation array time index is one earlier than the 
        # trajectory index.
        Y_obs_pert = np.repeat(Y_obs[:, index], n_ens).reshape(n_obs, n_ens) \
                     + obs_pert
    else:
        Y_obs_pert = np.repeat(Y_obs[:, index], n_ens).reshape(n_obs, n_ens)

    '''
        Step 2.(b)
        '''
    #### CALCULATE KALMAN GAIN, INNOVATIONS, AND ANALYSIS STATES ####
    Xbar = np.repeat(X.mean(axis=1), n_ens).reshape(n_d, n_ens)
    Xdev = X - Xbar # deviations

    # construct localisation matrix rho based on Gaspari Cohn function
    rho = gaspcohn_matrix(loc,Nk_fc,Neq)
    print('loc matrix rho shape: ', np.shape(rho))
    
    # compute innovation d = Y-H*X
    D = Y_obs_pert - np.matmul(H,X)

    # construct K
    R = np.repeat(ob_noise,[n_obs_h,n_obs_u,n_obs_r])*np.repeat(ob_noise,[n_obs_h,n_obs_u,n_obs_r])*np.identity(n_obs) # obs cov matrix
    HKd = np.empty([n_obs, n_ens]) 
    HKtr = np.empty(n_ens) 
    Xan = np.empty([n_d, n_ens]) 

    # analysis update
    for i in range(0,n_ens): 	 	
        Pf = np.matmul(np.delete(Xdev,i,1), np.delete(Xdev,i,1).T) 	 	
        Pf = Pf / (n_ens - 2)
        Ktemp = np.matmul(H, np.matmul(rho * Pf, H.T)) + R # H B H^T + R 	
        Ktemp = np.linalg.inv(Ktemp) # [H B H^T + R]^-1 	 	
        K = np.matmul(np.matmul(rho * Pf, H.T), Ktemp) # (rho Pf)H^T [H (rho Pf) H^T + R]^-1 	 	
        Xan[:,i] = X[:,i] + np.matmul(K,D[:,i]) 	 	
        HK = np.matmul(H,K) 	 	
        HKd[:,i] = np.diag(HK) 	 	
        HKtr[i] = np.trace(HK)

    # masks for locating model variables in state vector
    h_mask = list(range(0,Nk_fc))
    hu_mask = list(range(Nk_fc,2*Nk_fc))
    hr_mask = list(range(2*Nk_fc,3*Nk_fc))

    ### Relaxation to prior perturbations - Zhang et al. (2004)
    if rtpp != 0.0: # relax the ensemble
        print('RTPP factor =', rtpp)
        Pf = np.matmul(Xdev, Xdev.T) 	 	
        Pf = Pf / (n_ens - 1)
        Xanbar = np.repeat(Xan.mean(axis=1), n_ens).reshape(n_d, n_ens)
        Xandev = Xan - Xanbar  # analysis deviations
        Xandev = (1 - rtpp) * Xandev + rtpp * Xdev
        Xan = Xandev + Xanbar # relaxed analysis ensemble
    else:
        print('No RTPP relaxation applied')
    
    ### Relaxation to prior spread - Whitaker and Hamill (2012)
    if rtps != 0.0: # relax the ensemble
        print('RTPS factor =', rtps)
        Pf = np.matmul(Xdev, Xdev.T)           
        Pf = Pf / (n_ens - 1)
        sigma_b = np.sqrt(np.diagonal(Pf))
        Xanbar = np.repeat(Xan.mean(axis=1), n_ens).reshape(n_d, n_ens)
        Xandev = Xan - Xanbar  # analysis deviations
        Pa = np.matmul(Xandev, Xandev.T)               
        Pa = Pa / (n_ens - 1)
        sigma_a = np.sqrt(np.diagonal(Pa))
        alpha = 1 - rtps + rtps * sigma_b / sigma_a
        print("Min/max RTPS inflation factors = ", np.min(alpha), np.max(alpha))
        Xandev = Xandev * alpha[:, None]
        Xan = Xandev + Xanbar # relaxed analysis ensemble
    else:
        print('No RTPS relaxation applied')

    # if r < 0, set to zero:
    r = Xan[hr_mask,:]
    r[r < 0.] = 0.
    Xan[hr_mask,:] = r
    
    # if h < 0, set to epsilon:
    h = Xan[h_mask,:]
    h[h < 0.] = 1e-3
    Xan[h_mask,:] = h
    
    # transform from X to U for next integration (in h, hu, hr coordinates)
    U_an = np.empty((Neq,Nk_fc,n_ens))
    Xan[hu_mask,:] = Xan[hu_mask,:] * Xan[h_mask,:]
    Xan[hr_mask,:] = Xan[hr_mask,:] * Xan[h_mask,:]
    for N in range(0,n_ens):
        U_an[:,:,N] = Xan[:,N].reshape(Neq,Nk_fc)
    
    # now inflated, transform back to x = (h,u,r) for saving and later plotting
    Xan[hu_mask,:] = Xan[hu_mask,:] / Xan[h_mask,:]
    Xan[hr_mask,:] = Xan[hr_mask,:] / Xan[h_mask,:]
     
    ## observational influence diagnostics
    print(' ')
    print('--------- OBSERVATIONAL INFLUENCE DIAGNOSTICS:---------')
    print(' ')
    print(' Benchmark: global NWP has an average OI of ~0.18... ')
    print(' ... high-res. NWP less clear but should be 0.15 - 0.4')
    print('Check below: ')
    OI = np.mean(HKtr, 0) / n_obs
    HKd_mean = np.mean(HKd, 1)
    OI_h = np.sum(HKd_mean[h_obs_mask])/n_obs
    OI_hu = np.sum(HKd_mean[hu_obs_mask])/n_obs
    OI_hr = np.sum(HKd_mean[hr_obs_mask])/n_obs
    OI_vec = np.array([OI , OI_h , OI_hu, OI_hr])
    
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
 
    print('OI =', OI)
    print('OI check = ', np.sum(HKd_mean)/n_obs)
    print('OI_h =', OI_h)
    print('OI_hu =', OI_hu)
    print('OI_hr =', OI_hr)
    print(' ')
    print(' ')
    print('----------------------------------------------')
    print('------------- ANALYSIS STEP: END -------------')
    print('----------------------------------------------')
    print(' ')
    
    
    return U_an, U_fc, X, X_tr, Xan, OI_vec

# same analysis step as above but for 4 variables
def analysis_step_enkf_v4(U_fc, U_tr, Y_obs, H, tmeasure, dtmeasure, index, pars_enda, pars):
    '''
        (Steps refer to algorithm on page 121 of thesis, as do eq. numbers)
        
        INPUTS
        U_fc: ensemble trajectories in U space, shape (Neq,Nk_fc,n_ens)
        U_tr: truth trajectory in U space, shape (Neq,Nk_tr,Nmeas+1)
        tmeasure: time of assimilation
        dtmeasure: length of window
        pars_enda: vector of parameters relating to DA fixes (relaxation and localisation)
        '''
    
    print(' ')
    print('----------------------------------------------')
    print('------------ ANALYSIS STEP: START ------------')
    print('----------------------------------------------')
    print(' ')

    L = pars[14]
    Nk_fc = pars[0]
    n_ens = pars[2]
    Neq = pars[13]
    dres = pars[8]
    n_obs = pars[6]
    n_obs_h = pars[9]
    n_obs_u = pars[10]
    n_obs_r = pars[11]
    n_d = pars[12]  
    ob_noise = pars[15]
    n_obs_v = pars[16] 
    h_obs_mask = pars[17]
    hu_obs_mask = pars[18]
    hr_obs_mask = pars[19]
    if(Neq==4): hv_obs_mask = pars[20]

    #Nk_fc = np.shape(U_fc)[1] # fc resolution (no. of cells)
    Kk_fc = L/Nk_fc # fc resolution (cell length)
    rtpp = pars_enda[0] # relaxation factor
    rtps = pars_enda[1] # relaxation factor
    loc = pars_enda[2]
    add_inf = pars_enda[3]
    
    print(' ')
    print('--------- ANALYSIS: EnKF ---------')
    print(' ')
    print('Assimilation time = ', tmeasure)
    print('Number of ensembles = ', n_ens)
    
    # project truth onto forecast grid so that U and U_tr are the same dimension
    U_tmp = np.empty([Neq,Nk_fc])
    for i in range(0,Nk_fc):
        U_tmp[:,i] = U_tr[:, i*dres, index+1]
    U_tr = U_tmp

    '''
        step 1.(c)
        '''
    # for assimilation, work with [h,u,r]
    U_fc[1:,:,:] = U_fc[1:,:,:]/U_fc[0,:,:]
    U_tr[1:,:] = U_tr[1:,:]/U_tr[0,:]
    
    X_tr = U_tr.flatten()
    X_tr = X_tr.T
    
    # state matrix (flatten the array)
    X = np.empty([n_d,n_ens])
    for N in range(0,n_ens):
        X[:,N] = U_fc[:,:,N].flatten()
    

    '''
        Step 2.(a)
        '''
    # Add observation perturbations to each member, ensuring that the
    # observation perturbations have zero mean over all members to
    # avoid perturbing the ensemble mean. Do not apply when rtpp factor is 0.5
    # as Sakov and Oke (2008) results are equivalent to saying that rtpp 0.5
    # gives a deterministic ensemble Kalman filter in which perturbation
    # observations should not be applied.
    #print(ob_noise)
    #ob_noise = np.repeat(ob_noise, [n_obs_h,n_obs_u,n_obs_r])
    if rtpp != 0.5:
        obs_pert = np.repeat(ob_noise, [n_obs_h,n_obs_u,n_obs_r])[:, None] * np.random.randn(n_obs, n_ens)
        obs_pert_mean = np.mean(obs_pert, axis=1)
        obs_pert -= np.repeat(obs_pert_mean, n_ens).reshape(n_obs, n_ens)

        print('obs_pert shape =', np.shape(obs_pert))
    
        # y_o = y_m + e_o (eq. 6.6), with y_m itself a perturbed observation.
        # N.B. The observation array time index is one earlier than the 
        # trajectory index.
        Y_obs_pert = np.repeat(Y_obs[:, index], n_ens).reshape(n_obs, n_ens) \
                     + obs_pert
    else:
        Y_obs_pert = np.repeat(Y_obs[:, index], n_ens).reshape(n_obs, n_ens)

    '''
        Step 2.(b)
        '''
    #### CALCULATE KALMAN GAIN, INNOVATIONS, AND ANALYSIS STATES ####
    Xbar = np.repeat(X.mean(axis=1), n_ens).reshape(n_d, n_ens)
    Xdev = X - Xbar # deviations

    # construct localisation matrix rho based on Gaspari Cohn function
    rho = gaspcohn_matrix(loc,Nk_fc,Neq)
    print('loc matrix rho shape: ', np.shape(rho))
    
    # compute innovation d = Y-H*X
    D = Y_obs_pert - np.matmul(H,X)

    # construct K
    R = np.repeat(ob_noise,[n_obs_h,n_obs_u,n_obs_v,n_obs_r])*np.repeat(ob_noise,[n_obs_h,n_obs_u,n_obs_v,n_obs_r])*np.identity(n_obs) # obs cov matrix
    HKd = np.empty([n_obs, n_ens]) 
    HKtr = np.empty(n_ens) 
    Xan = np.empty([n_d, n_ens]) 

    # analysis update that takes inbreeding issue into account
    for i in range(0,n_ens): 	 	
        Pf = np.matmul(np.delete(Xdev,i,1), np.delete(Xdev,i,1).T) 	 	
        Pf = Pf / (n_ens - 2)
        Ktemp = np.matmul(H, np.matmul(rho * Pf, H.T)) + R # H B H^T + R 	
        Ktemp = np.linalg.inv(Ktemp) # [H B H^T + R]^-1 	 	
        K = np.matmul(np.matmul(rho * Pf, H.T), Ktemp) # (rho Pf)H^T [H (rho Pf) H^T + R]^-1 	 	
        Xan[:,i] = X[:,i] + np.matmul(K,D[:,i]) 	 	
        HK = np.matmul(H,K) 	 	
        HKd[:,i] = np.diag(HK) 	 	
        HKtr[i] = np.trace(HK)

    # masks for locating model variables in state vector
    h_mask = list(range(0,Nk_fc))
    hu_mask = list(range(Nk_fc,2*Nk_fc))
    hv_mask = list(range(2*Nk_fc,3*Nk_fc))
    hr_mask = list(range(3*Nk_fc,4*Nk_fc))

    ### Relaxation to prior perturbations - Zhang et al. (2004)
    if rtpp != 0.0: # relax the ensemble
        print('RTPP factor =', rtpp)
        Pf = np.matmul(Xdev, Xdev.T) 	 	
        Pf = Pf / (n_ens - 1)
        Xanbar = np.repeat(Xan.mean(axis=1), n_ens).reshape(n_d, n_ens)
        Xandev = Xan - Xanbar  # analysis deviations
        Xandev = (1 - rtpp) * Xandev + rtpp * Xdev
        Xan = Xandev + Xanbar # relaxed analysis ensemble
    else:
        print('No RTPP relaxation applied')
    
    ### Relaxation to prior spread - Whitaker and Hamill (2012)
    if rtps != 0.0: # relax the ensemble
        print('RTPS factor =', rtps)
        Pf = np.matmul(Xdev, Xdev.T)           
        Pf = Pf / (n_ens - 1)
        sigma_b = np.sqrt(np.diagonal(Pf))
        Xanbar = np.repeat(Xan.mean(axis=1), n_ens).reshape(n_d, n_ens)
        Xandev = Xan - Xanbar  # analysis deviations
        Pa = np.matmul(Xandev, Xandev.T)               
        Pa = Pa / (n_ens - 1)
        sigma_a = np.sqrt(np.diagonal(Pa))
        alpha = 1 - rtps + rtps * sigma_b / sigma_a
        print("Min/max RTPS inflation factors = ", np.min(alpha), np.max(alpha))
        Xandev = Xandev * alpha[:, None]
        Xan = Xandev + Xanbar # relaxed analysis ensemble
    else:
        print('No RTPS relaxation applied')

    # if r < 0, set to zero:
    r = Xan[hr_mask,:]
    r[r < 0.] = 0.
    Xan[hr_mask,:] = r
    
    # if h < 0, set to epsilon:
    h = Xan[h_mask,:]
    h[h < 0.] = 1e-3
    Xan[h_mask,:] = h
    
    # transform from X to U for next integration (in h, hu, hr coordinates)
    U_an = np.empty((Neq,Nk_fc,n_ens))
    Xan[hu_mask,:] = Xan[hu_mask,:] * Xan[h_mask,:]
    Xan[hv_mask,:] = Xan[hv_mask,:] * Xan[h_mask,:]
    Xan[hr_mask,:] = Xan[hr_mask,:] * Xan[h_mask,:]
    for N in range(0,n_ens):
        U_an[:,:,N] = Xan[:,N].reshape(Neq,Nk_fc)
    
    # now inflated, transform back to x = (h,u,r) for saving and later plotting
    Xan[hu_mask,:] = Xan[hu_mask,:] / Xan[h_mask,:]
    Xan[hv_mask,:] = Xan[hv_mask,:] / Xan[h_mask,:]
    Xan[hr_mask,:] = Xan[hr_mask,:] / Xan[h_mask,:]
    
    ## observational influence diagnostics
    print(' ')
    print('--------- OBSERVATIONAL INFLUENCE DIAGNOSTICS:---------')
    print(' ')
    print(' Benchmark: global NWP has an average OI of ~0.18... ')
    print(' ... high-res. NWP less clear but should be 0.15 - 0.4')
    print('Check below: ')
    OI = np.mean(HKtr, 0) / n_obs
    HKd_mean = np.mean(HKd, 1)
    OI_h = np.sum(HKd_mean[h_obs_mask])/n_obs
    OI_hu = np.sum(HKd_mean[hu_obs_mask])/n_obs
    OI_hv = np.sum(HKd_mean[hv_obs_mask])/n_obs
    OI_hr = np.sum(HKd_mean[hr_obs_mask])/n_obs
    OI_vec = np.array([OI , OI_h , OI_hu, OI_hv, OI_hr])
    
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
 
    print('OI =', OI)
    print('OI check = ', np.sum(HKd_mean)/n_obs)
    print('OI_h =', OI_h)
    print('OI_hu =', OI_hu)
    print('OI_hv =', OI_hv)
    print('OI_hr =', OI_hr)
    print(' ')
    print(' ')
    print('----------------------------------------------')
    print('------------- ANALYSIS STEP: END -------------')
    print('----------------------------------------------')
    print(' ')
    
    
    return U_an, U_fc, X, X_tr, Xan, OI_vec

