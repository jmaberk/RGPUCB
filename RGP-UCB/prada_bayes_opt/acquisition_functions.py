#from __future__ import division
#import sys
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.optimize import fsolve
#from prada_bayes_opt.acquisition_maximization import acq_max
import math

counter = 0

###############################################################################
class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq,delta=0.1):
        self.acq=acq
        acq_name=acq['name']
        
#        if bb_function:
#            self.bb_function=bb_function

        if 'WW' not in acq:
            self.WW=False
        else:
            self.WW=acq['WW']
        if 'WW_dim' not in acq:
            self.WW_dim=False
        else:
            self.WW_dim=acq['WW_dim']
        ListAcq=['ei','ucb','thompson','rucb_explore','rucb_balance','rucb_exploit']
        
        # check valid acquisition function
        IsTrue=[val for idx,val in enumerate(ListAcq) if val in acq_name]
        #if  not in acq_name:
        if  IsTrue == []:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, rucb_explore, rucb_balance, or rucb_exploit.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
            
        self.dim=acq['dim']
        
        if 'scalebounds' not in acq:
            self.scalebounds=[0,1]*self.dim
            
        else:
            self.scalebounds=acq['scalebounds']
        self.initialized_flag=0
        self.objects=[]
        self.delta=delta
        self.num_initial_points=acq['num_initial_points']
        self.max_iterations=acq['max_iterations']
        self.iterations_num=acq['iterations_num']

    def acq_kind(self, x, gp, y_max):

        #print self.kind
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'ei':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'ucb':
            return self._ucb(x, gp)
        if self.acq_name == 'rucb_exploit':
            return self._rucb_exploit(x, gp) 
        if self.acq_name == 'rucb_explore':
            return self._rucb_explore(x, gp) 
        if self.acq_name == 'rucb_balance':
            return self._rucb_balance(x, gp) 
        if self.acq_name == 'rucb_reverse':
            return self._rucb_reverse(x, gp)         
        if self.acq_name == 'ucb_reverse':
            return self._ucb_reverse(x, gp)         

    @staticmethod
    def _ei(x, gp, y_max):
        """
        Calculates the EI acquisition function values
        Inputs: gp: The Gaussian process, also contains all data
                y_max: The maxima of the found y values
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """
        y_max=np.asscalar(y_max)
        mean, var = gp.predict(x, eval_MSE=True)
        var2 = np.maximum(var, 1e-8 + 0 * var)
        z = (mean - y_max)/np.sqrt(var2)        
        out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-8]=0
        return out
    def _ucb(self,x, gp):
        """
        Calculates the GP-UCB acquisition function values
        Inputs: gp: The Gaussian process, also contains all data
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0 #prevents negative variances obtained through comp errors
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T  
        beta=2*np.log(np.power(len(gp.Y),self.dim/2+2)*np.square(math.pi)/(3*self.delta))
        #print("beta={}".format(beta))
        return mean +  np.sqrt(beta)* np.sqrt(var)


    def _rucb_exploit(self,x, gp):
        """
        Calculates the RGP-UCB acquisition function values with a low theta,
        favoring exploitation.
        Inputs: gp: The Gaussian process, also contains all data
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """

        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0 #prevents negative variances obtained through comp errors
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
        np.random.seed(len(gp.Y)*self.iterations_num)
        theta=0.1
        k=np.log((np.square(len(gp.Y))+1)/np.sqrt(2)*math.pi)/np.log(1+theta/2)
        distbeta=np.random.gamma(scale=k,shape=theta,size=1)         
        #print('k={}, theta={}'.format(k,theta))
        return mean +np.sqrt(distbeta)* np.sqrt(var)
	
    def _rucb_balance(self,x, gp):
        """
        Calculates the RGP-UCB acquisition function values with a theta
        favoring a balance of exploration and expoitation.
        Inputs: gp: The Gaussian process, also contains all data
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """

        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0 #prevents negative variances obtained through comp errors
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
        np.random.seed(len(gp.Y)*self.iterations_num)
        theta=1
        k=np.log((np.square(len(gp.Y))+1)/np.sqrt(2)*math.pi)/np.log(1+theta/2)
        distbeta=np.random.gamma(scale=k,shape=theta,size=1)         
        #print('k={}, theta={}'.format(k,theta))
        return mean +np.sqrt(distbeta)* np.sqrt(var)
	
    def _rucb_explore(self,x, gp):
        """
        Calculates the RGP-UCB acquisition function values with a high theta,
        favoring exploration.
        Inputs: gp: The Gaussian process, also contains all data
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """

        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0 #prevents negative variances obtained through comp errors
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
        np.random.seed(len(gp.Y)*self.iterations_num)
        theta=8
        k=np.log((np.square(len(gp.Y))+1)/np.sqrt(2)*math.pi)/np.log(1+theta/2)
        distbeta=np.random.gamma(scale=k,shape=theta,size=1)         
        #print('k={}, theta={}'.format(k,theta))
        return mean +np.sqrt(distbeta)* np.sqrt(var)
        
    class ThompsonSampling(object):
        """
        Calculates the Thompson sampling acquisition function values.
        Inputs: gp: The Gaussian process, also contains all data
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """
        #Calculates the thompson sample paramers 
        def __init__(self,gp,seed=False):
            var_mag=1
            ls_mag=1
            if seed!=False:
                np.random.seed(seed)
            dim=gp.X.shape[1]
            # used for Thompson Sampling
            self.WW_dim=100 # dimension of random feature
            self.WW=np.random.multivariate_normal([0]*self.WW_dim,np.eye(self.WW_dim),dim)/(gp.lengthscale*ls_mag)
            self.bias=np.random.uniform(0,2*3.14,self.WW_dim)

            # computing Phi(X)^T=[phi(x_1)....phi(x_n)]
            Phi_X=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(gp.X,self.WW)+self.bias), np.cos(np.dot(gp.X,self.WW)+self.bias)]) # [N x M]
            
            # computing A^-1
            A=np.dot(Phi_X.T,Phi_X)+np.eye(2*self.WW_dim)*gp.noise_delta*var_mag
            gx=np.dot(Phi_X.T,gp.Y)
            self.mean_theta_TS=np.linalg.solve(A,gx)
        #Calculates the thompson sample value at the point x    
        def __call__(self,x,gp):
            phi_x=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(x,self.WW)+self.bias), np.cos(np.dot(x,self.WW)+self.bias)])
            
            # compute the TS value
            gx=np.dot(phi_x,self.mean_theta_TS)    
            return gx
  
    
def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]



class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'
