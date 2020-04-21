# -*- coding: utf-8 -*-
'''
Name: pradda_gaussian_process.py
Authors: Julian Berk and Vu Nguyen
Publication date:15/01/2020
Classes and functions used for the Gaussian process statistical model.
'''

# define Gaussian Process class

from __future__ import division
import numpy as np
from prada_bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from scipy.optimize import minimize

from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
#from eucl_dist.cpu_dist import dist
from sklearn.cluster import KMeans
import scipy.linalg as spla
import math
import matplotlib.pyplot as plt



from scipy.spatial.distance import squareform

class PradaGaussianProcess(object):
    
    def __init__ (self,param):
        """
        Class for the Gaussian process
        
        Input parameters
        ----------
        
        param:                      Contains the various parameters for the GP model
        param.kernel_name :         Specifies the kernel being used
        param.lengthscale:          lengthscale used in the kernel
        param.noise_delta:          standard deviation of corrupting noise
        """
        kernel_name=param['kernel_name']
        if kernel_name not in ['SE','2_freq']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose SE or 2_freq.".format(kernel_name)
            raise NotImplementedError(err)
        else:
            self.kernel_name = kernel_name          
        if 'lengthscale' not in param:
            self.lengthscale=param['l']
        else:
            self.lengthscale=param['lengthscale']
        
        self.nGP=0
        # noise delta is for GP version with noise
        self.noise_delta=param['noise_delta']
        
        self.KK_x_x=[]
        self.KK_x_x_inv=[]
    
        self.X=[]
        self.Y=[]
        self.lengthscale_old=self.lengthscale
        self.flagOptimizeHyperFirst=0

    def kernel_dist(self, a,b,lengthscale):
        """
        Computes and returns the distance between points a and b for a given kernel and
        lengthscale
        
        """
        if self.kernel_name == 'ARD':
            return self.ARD_dist_func(a,b,lengthscale)
        if self.kernel_name=='SE':
            Euc_dist=euclidean_distances(a,b)
            return np.exp(-np.square(Euc_dist)/lengthscale)

            
    def fit(self,X,Y):
        """
        Fit Gaussian Process model

        Input Parameters
        ----------
        x: the observed points 
        y: the outcome y=f(x)
        
        """ 
        ur = unique_rows(X)
        X=X[ur]
        Y=Y[ur]
        
        self.X=X
        self.Y=Y
        
        if self.kernel_name=='SE':
            Euc_dist=euclidean_distances(X,X)
            self.KK_x_x=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(len(X))*self.noise_delta
        else:
            KK=pdist(self.X,lambda a,b: self.kernel_dist(a,b,self.lengthscale)) 
            KK=squareform(KK)
            self.KK_x_x=KK+np.eye(self.X.shape[0])*(1+self.noise_delta)
        
        if np.isnan(self.KK_x_x).any(): #NaN
            print("bug")
        self.KK_x_x_inv=np.linalg.pinv(self.KK_x_x)
        

    def log_marginal_lengthscale(self,lengthscale,noise_delta):
        """
        Compute Log Marginal likelihood of the GP model w.r.t. the provided lengthscale
        """

        def compute_log_marginal(X,lengthscale,noise_delta):
            # compute K
            ur = unique_rows(self.X)
            myX=self.X[ur]
            myY=self.Y[ur]
            if self.flagOptimizeHyperFirst==0:
                if self.kernel_name=='SE':
                    self.Euc_dist_X_X=euclidean_distances(myX,myX)
                    KK=np.exp(-np.square(self.Euc_dist_X_X)/(2*np.square(lengthscale)))+np.eye(len(myX))*self.noise_delta
                else:
                    KK=pdist(myX,lambda a,b: self.kernel_dist(a,b,lengthscale))
                    KK=squareform(KK)
                    KK=KK+np.eye(myX.shape[0])*(1+noise_delta)
                self.flagOptimizeHyperFirst=1
            else:
                if self.kernel_name=='SE':
                    KK=np.exp(-np.square(self.Euc_dist_X_X)/(2*np.square(lengthscale)))+np.eye(len(myX))*self.noise_delta
                else:
                    KK=pdist(myX,lambda a,b: self.kernel_dist(a,b,lengthscale))
                    KK=squareform(KK)
                    KK=KK+np.eye(myX.shape[0])*(1+noise_delta)

            try:
                temp_inv=np.linalg.solve(KK,np.eye(KK.shape[0]))
            except: # singular
                return -np.inf
            
            first_term=-0.5*np.dot(myY.T,np.dot(temp_inv,myY))
            
            # if the matrix is too large, we randomly select a part of the data for fast computation
            if KK.shape[0]>200:
                idx=np.random.permutation(KK.shape[0])
                idx=idx[:200]
                KK=KK[np.ix_(idx,idx)]
            chol  = spla.cholesky(KK, lower=True)
            W_logdet=np.sum(np.log(np.diag(chol)))
            # Uses the identity that log det A = log prod diag chol A = sum log diag chol A

            second_term=-0.5*W_logdet
            #print "first term ={:.4f} second term ={:.4f}".format(np.asscalar(first_term),np.asscalar(second_term))

            logmarginal=first_term+second_term-0.5*len(myY)*np.log(2*3.14)
            #print("lengthscale={}, likelihood={}".format(self.lengthscale,logmarginal))
            if np.isnan(np.asscalar(logmarginal))==True:
                print(myY)
                print("l={:s} first term ={:.4f} second  term ={:.4f}".format(lengthscale,np.asscalar(first_term),np.asscalar(second_term)))
                #print temp_det

            return np.asscalar(logmarginal)
        
        #print lengthscale
        logmarginal=0
        
        if np.isscalar(lengthscale):
            logmarginal=compute_log_marginal(self.X,lengthscale,noise_delta)
            return logmarginal

        if not isinstance(lengthscale,list) and len(lengthscale.shape)==2:
            logmarginal=[0]*lengthscale.shape[0]
            for idx in range(lengthscale.shape[0]):
                logmarginal[idx]=compute_log_marginal(self.X,lengthscale[idx],noise_delta)
        else:
            logmarginal=compute_log_marginal(self.X,lengthscale,noise_delta)
                
        #print logmarginal
        return logmarginal
    
    def optimize_lengthscale_SE(self,previous_l,noise_delta,bounds):
        """
        Optimize to select the optimal lengthscale parameter
        """
        dim=self.X.shape[1]
        scale_factor=1
        bounds_lengthscale_min=0.01*scale_factor
        print("scale_factor={}".format(scale_factor))
        bounds_lengthscale_max=0.4*scale_factor
        mybounds=[np.asarray([bounds_lengthscale_min,bounds_lengthscale_max]).T]
       
        
        lengthscale_tries = np.arange(bounds_lengthscale_min,bounds_lengthscale_max,0.001).reshape(-1,1)    

        # evaluate
        self.flagOptimizeHyperFirst=0 # for efficiency

        logmarginal_tries=self.log_marginal_lengthscale(lengthscale_tries,noise_delta)
        print("Y.size={}".format(self.Y.size))
        TS=AcquisitionFunction.ThompsonSampling(self)
        x=np.arange(0,1,0.01).reshape(-1,1)
        g=TS(x,self)
        idx_max=np.argmax(logmarginal_tries)
        lengthscale_init_max=lengthscale_tries[idx_max]
        #print lengthscale_init_max
        
        myopts ={'maxiter':10,'maxfun':10}

        x_max=[]
        max_log_marginal=None
        
        for i in range(dim):
            res = minimize(lambda x: -self.log_marginal_lengthscale(x,noise_delta),lengthscale_init_max,
                           bounds=mybounds,method="L-BFGS-B",options=myopts)#L-BFGS-B
            if 'x' not in res:
                val=self.log_marginal_lengthscale(res,noise_delta)    
            else:
                val=self.log_marginal_lengthscale(res.x,noise_delta)  

            # Store it if better than previous minimum(maximum).
            if max_log_marginal is None or val >= max_log_marginal:
                if 'x' not in res:
                    x_max = res
                else:
                    x_max = res.x
                max_log_marginal = val
            #print res.x
        return x_max

    def optimize_lengthscale(self,previous_l,noise_delta,bounds):
        if self.kernel_name == 'ARD':
            return self.optimize_lengthscale_ARD(previous_l,noise_delta)
        if self.kernel_name=='SE':
            return self.optimize_lengthscale_SE(previous_l,noise_delta,bounds)


    def compute_var(self,X,xTest):
        """
        compute variance given X and xTest
        
        Input Parameters
        ----------
        X: the observed points
        xTest: the testing points 
        
        Returns
        -------
        diag(var)
        """ 
        
        xTest=np.asarray(xTest)
        if self.kernel_name=='SE':
            ur = unique_rows(X)
            X=X[ur]
            if xTest.shape[0]<300:
                Euc_dist_test_train=euclidean_distances(xTest,X)
                #Euc_dist_test_train=dist(xTest, X, matmul='gemm', method='ext', precision='float32')
                KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/self.lengthscale)
            else:
                KK_xTest_xTrain=cdist(xTest,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))

            Euc_dist_train_train=euclidean_distances(X,X)
            self.KK_bucb_train_train=np.exp(-np.square(Euc_dist_train_train)/self.lengthscale)+np.eye(X.shape[0])*self.noise_delta        
        else:
            ur = unique_rows(X)
            X=X[ur]
            KK_xTest_xTrain=cdist(xTest,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
            self.KK_bucb_train_train=cdist(X,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))+np.eye(X.shape[0])*self.noise_delta
        try:
            temp=np.linalg.solve(self.KK_bucb_train_train,KK_xTest_xTrain.T)
        except:
            temp=np.linalg.lstsq(self.KK_bucb_train_train,KK_xTest_xTrain.T, rcond=-1)
            temp=temp[0]
            
        var=np.eye(xTest.shape[0])-np.dot(temp.T,KK_xTest_xTrain.T)
        var=np.diag(var)
        var.flags['WRITEABLE']=True
        var[var<1e-100]=0
        return var 

        
    def predict(self,xTest,eval_MSE=True):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    
        if len(xTest.shape)==1: # 1d
            xTest=np.array(xTest.reshape((-1,self.X.shape[1])))
        
        # prevent singular matrix
        ur = unique_rows(self.X)
        X=self.X[ur]
        Y=self.Y[ur]
        
        if self.kernel_name=='SE':
            Euc_dist=euclidean_distances(xTest,xTest)
            KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(xTest.shape[0])*self.noise_delta
            
            Euc_dist_test_train=euclidean_distances(xTest,X)
            KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/self.lengthscale)
        else:
            KK=pdist(xTest,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
            KK=squareform(KK)
            KK_xTest_xTest=KK+np.eye(xTest.shape[0])+np.eye(xTest.shape[0])*self.noise_delta
            KK_xTest_xTrain=cdist(xTest,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
        
        temp=np.dot(KK_xTest_xTrain,self.KK_x_x_inv)
        mean=np.dot(temp,self.Y)
        var=KK_xTest_xTest-np.dot(temp,KK_xTest_xTrain.T)

        return mean.ravel(),np.diag(var)  


    def posterior(self,x):
        # compute mean function and covariance function
        return self.predict(self,x)
        