# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""

from __future__ import division
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from prada_bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
#from visualization import Visualization
from prada_bayes_opt.prada_gaussian_process import PradaGaussianProcess
#from prada_gaussian_process import PradaMultipleGaussianProcess

from prada_bayes_opt.acquisition_maximization import acq_max
from prada_bayes_opt.acquisition_maximization import acq_max_thompson
from prada_bayes_opt.acquisition_maximization import acq_max_global
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy import optimize
from scipy import stats
from pyDOE import lhs
import matplotlib.pyplot as plt
from cycler import cycler
import time
import math


#@author: Julian

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
counter = 0

###############################################################################
class PradaBayOptFn(object):

    def __init__(self, gp_params, func_params, acq_params, experiment_num, seed):
        """      
        Input parameters
        ----------
        
        gp_params:                  GP parameters
        gp_params.l:                to compute the kernel
        gp_params.theta:            paramater for DGP-UCB gamma distribution
        gp_params.delta:            to compute the kernel
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:        bounds on parameters        
        func_params.func:           a function to be optimized
        
        
        acq_params:            acquisition function, 
        acq_params.acq_func['name']=['ei','ucb','poi','lei']
                            ,acq['kappa'] for ucb, acq['k'] for lei
        acq_params.opt_toolbox:     optimization toolbox 'nlopt','direct','scipy'
        
        experiment_num: the interation of the GP method. Used to make sure each 
                        independant stage of the experiment uses different 
                        initial conditions
        seed: Variable used as part of a seed to generate random initial points
                            
        Returns
        -------
        dim:            dimension
        scalebounds:    bound used thoughout the BO algorithm
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        self.experiment_num=experiment_num
        np.random.seed(self.experiment_num*seed)
        self.seed=seed
        
        # Prior distribution paramaters for the DDB method
        self.theta=1
        # Find number of parameters
        bounds=func_params['bounds']
        if 'init_bounds' not in func_params:
            init_bounds=bounds
        else:
            init_bounds=func_params['init_bounds']
        # Find input dimention
        self.dim = len(bounds)
        self.radius=np.ones([self.dim,1])

        # Generate bound array
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        # find function to be optimized
        self.f = func_params['f']

        # acquisition function type
        
        self.acq=acq_params['acq_func']
        self.delta=acq_params["delta"]
        self.acq['max_iterations']=acq_params['max_iterations']
        self.acq['num_initial_points']=acq_params['num_initial_points']
        self.acq['iterations_num']=acq_params['iterations_num']
        
        # Other checks
        if 'debug' not in self.acq:
            self.acq['debug']=0           
        if 'stopping' not in acq_params:
            self.stopping_criteria=0
        else:
            self.stopping_criteria=acq_params['stopping']
        if 'optimize_gp' not in acq_params:
            self.optimize_gp=0
        else:                
            self.optimize_gp=acq_params['optimize_gp']
        if 'marginalize_gp' not in acq_params:
            self.marginalize_gp=0
        else:                
            self.marginalize_gp=acq_params['marginalize_gp']
        
        # optimization toolbox
        if 'opt_toolbox' not in acq_params:
            if self.acq['name']=='ei_reg':
                self.opt_toolbox='unbounded'
            else:
                self.opt_toolbox='scipy'
        else:
            self.opt_toolbox=acq_params['opt_toolbox']
        self.iteration_factor=acq_params['iteration_factor']
        # store X in original scale
        self.X_original= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None
        
        # value of the acquisition function at the selected point
        self.alpha_Xt=None
        self.Tau_Xt=None
        
        self.time_opt=0

        self.k_Neighbor=2
        
        # Gaussian Process class
        self.gp=PradaGaussianProcess(gp_params)
        self.gp_params=gp_params
        #self.gp.theta=gp_params['theta']
        # acquisition function
        self.acq_func = None
    
        # stop condition
        self.stop_flag=0
        self.logmarginal=0
        
        # xt_suggestion, caching for Consensus
        self.xstars=[]
        self.ystars=np.zeros((2,1))
        
        # l vector for marginalization GP
        self.l_vector =[]
    
    def init(self,gp_params, n_init_points=3):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """
        # set seed to allow for reproducible results
        np.random.seed(self.experiment_num*self.seed)
        print(self.experiment_num)
        #Generate initial points on grid
        l=np.zeros([n_init_points,self.dim])
        bound_length=self.scalebounds[0,1]-self.scalebounds[0,0]
        for d in range(0,self.dim):
            l[:,d]=lhs(n_init_points)[:,0]
        self.X=np.asarray(l)+self.scalebounds[:,0]         
        self.X=self.X*bound_length #initial inouts
        print("starting points={}".format(self.X))
        y_init=self.f(self.X)
        y_init=np.reshape(y_init,(n_init_points,1))
        self.Y_original = np.asarray(y_init)     #initial outputs   
        print('initial_bound={}'.format(self.scalebounds))
        
    def maximize(self,gp_params):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.scalebounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            
            self.time_opt=np.hstack((self.time_opt,0))
            return         

        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        if self.gp.KK_x_x_inv ==[]:
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

 
        acq=self.acq
        self.acq_func = AcquisitionFunction(self.acq,self.delta)
        if acq['debug']==1:
            logmarginal=self.gp.log_marginal_lengthscale(gp_params['l'],gp_params['noise_delta'])
            print(gp_params['l'])
            print("log marginal before optimizing ={:.4f}".format(logmarginal))
            self.logmarginal=logmarginal
                
            if logmarginal<-999999:
                logmarginal=self.gp.log_marginal_lengthscale(gp_params['l'],gp_params['noise_delta'])

        if self.optimize_gp==1 and len(self.Y)%2*self.dim==0 and len(self.Y)>5*self.dim:

            print("Initial length scale={}".format(gp_params['l']))
            newl = self.gp.optimize_lengthscale(gp_params['l'],gp_params['noise_delta'],self.scalebounds)
            gp_params['l']=newl
            print("New length scale={}".format(gp_params['l']))

            # init a new Gaussian Process after optimizing hyper-parameter
            self.gp=PradaGaussianProcess(gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
        
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()        
                      
        if 'xstars' not in globals():
            xstars=[]
            
        self.xstars=xstars

        self.acq['xstars']=xstars
        self.acq['WW']=False
        self.acq['WW_dim']=False
        self.acq_func = AcquisitionFunction(self.acq,self.delta)

        if acq['name']=="thompson":
            x_max = acq_max_thompson(gp=self.gp,y_max=y_max,bounds=self.scalebounds)
        else:
            x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)
        
       

            
        val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)
        #print x_max
        #print val_acq
        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)

            self.stop_flag=1
            print("Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria))
        
        
        self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # evaluate Y using original X
        self.Y_original = np.append(self.Y_original, self.f(x_max))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)

        self.experiment_num=self.experiment_num+1