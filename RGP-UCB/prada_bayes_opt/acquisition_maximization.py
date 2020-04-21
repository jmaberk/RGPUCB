# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:51:41 2016

@author: Vu
"""
from __future__ import division
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics.pairwise import euclidean_distances
from prada_bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from scipy.optimize import fmin_cobyla

import random
import time


__author__ = 'Vu'

    
def acq_max(ac, gp, y_max, bounds, opt_toolbox='scipy',seeds=[]):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    if opt_toolbox=='scipy':
        x_max = acq_max_scipy(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    return x_max
        

def acq_max_scipy(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    myopts ={'maxiter':5*dim,'maxfun':10*dim}


    # multi start
    for i in range(2*dim):
        # Find the minimum of minus the acquisition function
      
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))
    
        # evaluate
        start_eval=time.time()
        y_tries=ac(x_tries,gp=gp, y_max=y_max)
        end_eval=time.time()
        #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
        start_opt=time.time()
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B   
        
        if 'x' not in res:
            val=ac(res,gp,y_max)        
        else:
            val=ac(res.x,gp,y_max) 

        
        end_opt=time.time()
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max
    
    
def acq_max_thompson(gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    myopts ={'maxiter':5*dim,'maxfun':10*dim}


    # multi start
    for i in range(2*dim):
        # Find the minimum of minus the acquisition function
        TS=AcquisitionFunction.ThompsonSampling(gp,seed=False)
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))
    
        # evaluate
        start_eval=time.time()
        y_tries=TS(x_tries,gp=gp)
        end_eval=time.time()
        #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
        start_opt=time.time()
        
        res = minimize(lambda x: -TS(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B   
        
        if 'x' not in res:
            val=TS(res,gp)        
        else:
            val=TS(res.x,gp) 

        
        end_opt=time.time()
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max
    
def acq_max_global(ac, gp, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None
        

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    myopts ={'maxiter':5*dim,'maxfun':10*dim}



    # multi start
    for i in xrange(1*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(60*dim, dim))
        # evaluate
        y_tries=ac(x_tries,gp)
        #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res,gp)        
        else:
            val=ac(res.x,gp) 

        
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq
        if res.fun==0:
            y_max=ac(x_max,gp)
        else:
            y_max=res.fun[0]
    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq

    return np.clip(x_max, bounds[:, 0], bounds[:, 1]), y_max
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max