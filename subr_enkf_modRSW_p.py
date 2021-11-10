#######################################################################
# Perturbed obs EnKF for modRSW with topography
#######################################################################

'''
May 2017

SUBROUTINE (p) for batch-processing EnKF experiments. 
Given parameters and truth run supplied by <main>, the function <run_enkf> carries out ensemble integratiopns 
IN PARALLEL using the multiprocessing module.

'''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno
import multiprocessing as mp
from datetime import datetime
import shutil
import importlib.util
import sys

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

#from parameters import * # module storing fixed parameters
from f_modRSW import make_grid, step_forward_topog, ens_forecast, ens_forecast_topog
from f_enkf_modRSW import analysis_step_enkf_v4
from create_readme import create_readme

def run_enkf(i, j, k, l, ic, U_tr_array, Y_obs, H, dirname, config_file):

    ################################################################
    # IMPORT PARAMETERS FROM CONFIGURATION FILE
    ################################################################

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config) 

    outdir = config.outdir
    Nk_tr = config.Nk_tr
    Nk_fc = config.Nk_fc
    L = config.L
    Neq = config.Neq
    dres = config.dres
    H0 = config.H0
    A = config.A
    V = config.V
    Ro = config.Ro
    Fr = config.Fr
    Hc = config.Hc
    Hr = config.Hr
    alpha2 = config.alpha2
    beta = config.beta
    cc2 = config.cc2
    g = config.g
    cfl_fc = config.cfl_fc
    cfl_tr = config.cfl_tr
    n_ens = config.n_ens
    TIMEOUT_PAR = config.TIMEOUT_PAR
    sig_ic = config.sig_ic
    dtmeasure = config.dtmeasure
    Nmeas = config.Nmeas
    Nforec = config.Nforec
    t_end_assim = config.t_end_assim
    NIAU = config.NIAU
    n_obs_h = config.n_obs_h
    n_obs_u = config.n_obs_u
    n_obs_r = config.n_obs_r
    if(Neq==4): n_obs_v = config.n_obs_v
    n_obs = config.n_obs
    ob_noise = config.ob_noise
    assim_time = config.assim_time
    loc = config.loc
    add_inf = config.add_inf
    rtpp = config.rtpp
    rtps = config.rtps
    U_relax = config.U_relax
    tau_rel = config.tau_rel

    ################################################################ 

    print(' ')
    print('---------------------------------------------------')
    print('----------------- EXPERIMENT '+str(i+1)+str(j+1)+str(k+1)+str(l+1) +'------------------')
    print('---------------------------------------------------')
    print(' ')

    pars_enda = [rtpp[k], rtps[l], loc[i], add_inf[j]]
    
    #################################################################
    # create directory for output
    #################################################################
    dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(dirname, str(loc[i]),
                                                     str(add_inf[j]), 
                                                     str(rtpp[k]),
                                                     str(rtps[l]))
    # Delete output directory if it exists and (re)create the output directory.
    try:
        if os.path.isdir(dirn):
            shutil.rmtree(dirn)
        os.makedirs(dirn)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    ##################################################################
    # Construct the model error covariance matrix
    ##################################################################
    Q = np.load(str(dirname + '/Qnew.npy'))
#    Q = np.diag(np.concatenate(map(lambda x: np.repeat(x**2, Nk_fc), model_noise)))

    ##################################################################
    # Mesh generation for forecasts
    ##################################################################

    fc_grid =  make_grid(Nk_fc,L) # forecast

    Kk_fc = fc_grid[0]
    x_fc = fc_grid[1]
    xc_fc = fc_grid[2]

    ##################################################################    
    #  Apply initial conditions
    ##################################################################
    print(' ') 
    print('---------------------------------------------------') 
    print('---------      ICs: generate ensemble     ---------')
    print('---------------------------------------------------') 
    print(' ') 
    print('Initial condition =', str(ic), '(see <init_cond_modRSW.py> for deatils ... )')
    print(' ') 
    ### Forecast ic 
    U0_fc, B = ic(x_fc,Nk_fc,Neq,H0,L,A,V) # control IC to be perturbed

    U0ens = np.empty([Neq,Nk_fc,n_ens])

    print('Initial ensemble perurbations:')
    print('sig_ic = [sig_h, sig_hu, sig_hr] =', sig_ic)    
    
    # Generate initial ensemble
    for jj in range(0,Neq):
        for N in range(0,n_ens):
            # add sig_ic to EACH GRIDPOINT
            U0ens[jj,:,N] = U0_fc[jj,:] + sig_ic[jj]*np.random.randn(Nk_fc)
            # add sig_ic to TRAJECTORY as a whole
            #U0ens[jj,:,N] = U0_fc[jj,:] + sig_ic[jj]*np.random.randn(1)

    if(Neq==3):
        # if hr < 0, set to zero:
        hr = U0ens[2, :, :]
        hr[hr < 0.] = 0.
        U0ens[2, :, :] = hr

    if(Neq==4):
        # if hr < 0, set to zero:
        hr = U0ens[3, :, :]
        hr[hr < 0.] = 0.
        U0ens[3, :, :] = hr

    # if h < 0, set to epsilon:
    h = U0ens[0, :, :]
    h[h < 0.] = 1e-3
    U0ens[0, :, :] = h
    
    np.save(dirn+'/U0ens.npy',U0ens)

    ### Relaxation solution ###
    U_rel = U_relax(Neq,Nk_fc,L,V,xc_fc,U0_fc)

    ##################################################################
    #%%%-----        Define arrays for outputting data       ------%%%
    ##################################################################
    nd = Neq*Nk_fc                          # total # d. of f.
    X_array = np.empty([nd,n_ens,Nmeas])
    X_array.fill(np.nan)
    Xan_array = np.empty([nd,n_ens,Nmeas])
    Xan_array.fill(np.nan)
    X_tr_array = np.empty([nd,1,Nmeas])
    X_tr_array.fill(np.nan)
    X_forec = np.empty([nd,n_ens,Nmeas,Nforec])
    q2store = np.empty([nd,n_ens,Nmeas,Nforec])
#    Y_obs_array = np.empty([n_obs,n_ens,Nmeas+1])
#    Y_obs_array.fill(np.nan)
    OI = np.empty([Neq+1,Nmeas])
    OI.fill(np.nan)

    # create readme file of exp summary and save
    if(Neq==3): PARS = [Nk_fc, Nk_tr, n_ens, assim_time, pars_enda, sig_ic, n_obs, NIAU, dres, n_obs_h, n_obs_u, n_obs_r, nd, Neq, L, ob_noise]
    if(Neq==4): PARS = [Nk_fc, Nk_tr, n_ens, assim_time, pars_enda, sig_ic, n_obs, NIAU, dres, n_obs_h, n_obs_u, n_obs_r, nd, Neq, L, ob_noise, n_obs_v]
    create_readme(dirn, config_file, i, j, k, l)
    
    ##################################################################
    #  Integrate ensembles forward in time until obs. is available   #
    ##################################################################
    print(' ')
    print('-------------------------------------------------')
    print('------ CYCLED FORECAST-ASSIMILATION SYSTEM ------')
    print('-------------------------------------------------')
    print('--------- ENSEMBLE FORECASTS + EnKF--------------')
    print('-------------------------------------------------')
    print(' ')
    
    # Initialise...
    U = U0ens
    #OI[:,0] = 0.0 # No obs assimilated at the initial time
    index = 0 # to step through assim_time
    tmeasure = dtmeasure # reset tmeasure    
    
    while tmeasure-dtmeasure < t_end_assim and index < Nmeas:
        
        os.sched_setaffinity(0, range(0, os.cpu_count()))
        print(np.shape(U))
     
        try: 
            if index==0:  
            
                print(' ')
                print('----------------------------------------------')
                print('------------ FORECAST STEP: START ------------')
                print('----------------------------------------------')
                print(' ')
            
                num_cores_use = os.cpu_count() 
           
                print('Starting ensemble integrations from time =', assim_time[index],' to',assim_time[index+1])
                print('Number of cores used:', num_cores_use)
                print(' *** Started: ', str(datetime.now()))

                ### ADDITIVE INFLATION (moved to precede the forecast) ###
                q = add_inf[j] * np.random.multivariate_normal(np.zeros(nd), Q, n_ens)
                q_ave = np.mean(q,axis=0)
                q = q - q_ave
                q = q.T

                q2store[:,:,index,0] = np.copy(q) 

                pool = mp.Pool(processes=num_cores_use)
                mp_out = [pool.apply_async(ens_forecast, args=(N, U, U_rel, q, Neq, Nk_fc, Kk_fc, cfl_fc, assim_time, index, tmeasure, dtmeasure, Hc, Hr, cc2, beta, alpha2, g, Ro, tau_rel)) for N in range(0,n_ens)]
                #mp_out = [pool.apply_async(ens_forecast_topog, args=(N, U, B, q, Neq, Nk_fc, Kk_fc, cfl_fc, assim_time, index, tmeasure, dtmeasure, Hc, Hr, cc2, beta, alpha2, g)) for N in range(0,n_ens)]
                U = [p.get(timeout=TIMEOUT_PAR) for p in mp_out]

                pool.close()
                pool.join()

                print(' All ensembles integrated forward from time =', assim_time[index],' to',assim_time[index+1])
                print(' *** Ended: ', str(datetime.now()))
                print(np.shape(U))

                U=np.swapaxes(U,0,1)
                U=np.swapaxes(U,1,2)
   
                print(np.shape(U))
        
                dU = np.copy(U)
                dU = U - np.repeat(U_tr_array[:,0::dres,index+1], n_ens).reshape([Neq, Nk_fc, n_ens])
                print("Forecast", dU[0].min(), dU[0].max(), dU[1].min(), dU[1].max(), dU[2].min(), dU[2].max())
                print(' ')
                print('----------------------------------------------')
                print('------------- FORECAST STEP: END -------------')
                print('----------------------------------------------')
                print(' ')
        
            ##################################################################
            #  calculate analysis at observing time then integrate forward  #
            ##################################################################
            if(Neq==4): U_an, U_fc, X_array[:,:,index], X_tr_array[:,0,index], Xan_array[:,:,index], OI[:,index] = analysis_step_enkf_v4(U, U_tr_array, Y_obs, H, tmeasure, dtmeasure, index, pars_enda, PARS)
            if(Neq==3): U_an, U_fc, X_array[:,:,index], X_tr_array[:,0,index], Xan_array[:,:,index], OI[:,index] = analysis_step_enkf_v3(U, U_tr_array, Y_obs, H, tmeasure, dtmeasure, index, pars_enda, PARS)
            U = U_an # update U with analysis ensembles for next integration
            dU = U_an - np.repeat(U_tr_array[:,0::dres,index+1], n_ens).reshape([Neq, Nk_fc, n_ens])
            print("Analysis", dU[0].min(), dU[0].max(), dU[1].min(), dU[1].max(), dU[2].min(), dU[2].max())
                
            #######################################
            #  generate a *Nforec*-long forecast # 
            #######################################

            print(' Long-range forecast starting... ')
                
            tforec = tmeasure
            tendforec = tforec + Nforec*dtmeasure
            forec_time = np.linspace(tforec,tendforec,Nforec+1)
            forec_T = 1
            U_forec = np.copy(U_an)

            X_forec[:,:,index,0] = np.copy(X_array[:,:,index])

            while tforec < tendforec and forec_T < Nforec: 

                ### ADDITIVE INFLATION (moved to precede the forecast) ###
                q = add_inf[j] * np.random.multivariate_normal(np.zeros(nd), Q, n_ens)
                q_ave = np.mean(q,axis=0)
                q = q - q_ave
                q = q.T
  
                if forec_T > NIAU: q[:,:] = 0.0
                
                q2store[:,:,index,forec_T] = np.copy(q)

                pool = mp.Pool(processes=num_cores_use)

                mp_out = [pool.apply_async(ens_forecast, args=(N, U_forec, U_rel, q, Neq, Nk_fc, Kk_fc, cfl_fc, forec_time, forec_T-1, tforec+dtmeasure, dtmeasure, Hc, Hr, cc2, beta, alpha2, g, Ro, tau_rel)) for N in range(0,n_ens)]
                #mp_out = [pool.apply_async(ens_forecast_topog, args=(N, U_forec, B, q, Neq, Nk_fc, Kk_fc, cfl_fc, forec_time, forec_T-1, tforec+dtmeasure, dtmeasure, Hc, Hr, cc2, beta, alpha2, g)) for N in range(0,n_ens)]
                U_forec = [p.get(timeout=TIMEOUT_PAR) for p in mp_out]

                pool.close()
                pool.join()

                U_forec = np.swapaxes(U_forec,0,1)
                U_forec = np.swapaxes(U_forec,1,2)

                if forec_T==1: U = np.copy(U_forec)

                U_forec_tmp = np.copy(U_forec)
                U_forec_tmp[1:,:,:] = U_forec_tmp[1:,:,:]/U_forec_tmp[0,:,:]
 
                for N in range(n_ens):
                    X_forec[:,N,index,forec_T] = U_forec_tmp[:,:,N].flatten()

                print(' All ensembles integrated forward from time =', round(tforec,3) ,' to', round(tforec+dtmeasure,3))
                
                tforec = tforec+dtmeasure
                forec_T = forec_T + 1

            # on to next assim_time
            index = index + 1
            tmeasure = tmeasure + dtmeasure

            print(' ANALYSIS STEP restarting at time =', round(tmeasure,3))
            
        except (RuntimeWarning, mp.TimeoutError) as err:
            pool.terminate()
            pool.join()
            print(err)
            print('-------------- Forecast failed! --------------')        
            print(' ')
            print('----------------------------------------------')
            print('------------- FORECAST STEP: END -------------')
            print('----------------------------------------------')
            print(' ')

            tmeasure = t_end_assim + dtmeasure
        

    ##################################################################


    #PARS = [Nk_fc, Nk_tr, n_ens, assim_time, pars_enda, sig_ic, n_obs, NIAU]

    # create readme file and save
    #create_readme(dirn, config_file)

    np.save(str(dirn+'/B'),B)
    np.save(str(dirn+'/X_array'),X_array)
    np.save(str(dirn+'/X_tr_array'),X_tr_array)
    np.save(str(dirn+'/Xan_array'),Xan_array)
    np.save(str(dirn+'/X_forec'),X_forec)
    np.save(str(dirn+'/qstored'),q2store)
#    np.save(str(dirn+'/Y_obs_array'),Y_obs_array)
    np.save(str(dirn+'/OI'),OI)
        
    print(' *** Data saved in :', dirn)
    print(' ')

    # print summary to terminal aswell
    print(' ') 
    print('---------------------------------------') 
    print('--------- END OF ASSIMILATION ---------') 
    print('---------------------------------------') 
    print(' ')   
    print(' -------------- SUMMARY: ------------- ')  
    print(' ') 
    print('Dynamics:')
    print('Ro =', Ro)  
    print('Fr = ', Fr)
    print('(H_0 , H_c , H_r) =', [H0, Hc, Hr]) 
    print('(alpha, beta, c2) = ', [alpha2, beta, cc2])
    print('[cfl_fc, cfl_tr] = ', [cfl_fc, cfl_tr])
    print('Initial condition =', str(ic))
    print(' ') 
    print('Assimilation:')
    print('Forecast resolution (number of gridcells) =', Nk_fc)
    print('Truth resolution (number of gridcells) =', Nk_tr)   
    if Nk_fc == Nk_tr: # perfect model
        print('>>> perfect model scenario')
    else:
        print('>>> imperfect model scenario') 
    print(' ')  
    print('Number of ensembles =', n_ens)  
#    print 'Assimilation times  =', assim_time[1:]
#    print 'Observation density: observe every', pars_ob[0], 'gridcells...'
    print('i.e., total no. of obs. =', n_obs)
    print('Observation noise =', ob_noise)  
    print('RTPP (ensemble) factor =', pars_enda[0])
    print('RTPS (ensemble) factor =', pars_enda[1])
    print('Additive inflation factor =', pars_enda[3])
    print('Localisation lengthscale =', pars_enda[2])
    print(' ')   
    print(' ----------- END OF SUMMARY: ---------- ')  
    print(' ')  





    ##################################################################
    #                       END OF PROGRAM                           #
    ##################################################################


