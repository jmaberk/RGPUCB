'''
Name: functions.py
Authors: Julian Berk and Vu Nguyen
Publication date:08/04/2019
Description: A range of benchmark and synthetic funtions to optimize with 
Bayesian optimization
'''

import numpy as np
from collections import OrderedDict
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

max_bound_size=1
###############################################################################
def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
    
class functions:
    def findExtrema(self):
        """
        Description: Estimates and prints the maxima and minima of a function 
        though brute force evaluation. Used for method evaluation. 
        Useage: Mucst be called from a function in the form 
        random_function.findExtrema().
        Output: Prints out the maxiam and minima of the desired function.
        """
        step_size=0.01
        #determine function bounds
        bounds=self.bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            keys = bounds.keys()
            arr_bounds = []
            for key in keys:
                arr_bounds.append(bounds[key])
            arr_bounds = np.asarray(arr_bounds)
        else:
            arr_bounds=np.asarray(bounds)
        #randomly generate points and evaluate them
        X=np.array([np.arange(x[0], x[1], step_size) for x in arr_bounds])
        y=self.func(X.T)
        #find optima
        functionMax=np.max(y)
        functionMin=np.min(y)
        print("function max={}".format(functionMax))
        print("function min={}".format(functionMin))
        return (functionMin,functionMax)
        
    def findSdev(self):
        """
        Description: Estimates the standard deviation of the function through
        multiple evaluations. Used to generate noise with appropriate
        magnitudes
        Useage: Mucst be called from a function in the form 
        random_function.findSdev().
        Output: An estimate of the functions standard deviation, sdv
        """
        num_points_per_dim=100
        #determine function bounds
        bounds=self.bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            keys = bounds.keys()
            arr_bounds = []
            for key in keys:
                arr_bounds.append(bounds[key])
            arr_bounds = np.asarray(arr_bounds)
        else:
            arr_bounds=np.asarray(bounds)
        #randomly generate points and evaluate them
        X=np.array([np.random.uniform(x[0], x[1], size=num_points_per_dim) for x in arr_bounds])
        X=X.reshape(num_points_per_dim,-1)
        y=self.func(X)
        #calculate standard deviation
        sdv=np.std(y)
        return sdv
    
class saddlepoint(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=saddlepoint() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: The function value at x
    """
    def __init__(self):
        self.input_dim=2
        self.bounds=OrderedDict({'x1':(-1,1),'x2':(-1,1)})
        self.fmin=0
        self.min=0
        self.ismax=1
        self.name='saddlepoint'
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]

        fval=X[:,0]*X[:,0]-X[:,1]*X[:,1]
        return fval*self.ismax
        
        
class sincos(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=sincos() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: The function value at x
    """
    def __init__(self,f1=1,f2=0.1):
        self.input_dim=1
        self.bounds={'x':(-1,2)}
        self.f1=f1
        self.f2=f2
        self.fmin=11
        self.min=0
        self.ismax=1
        self.name='sincos'
    def func(self,x):
        x=np.asarray(x)
        fval=np.sin(self.f1*x)+np.cos(self.f2*x)
        return fval*self.ismax
    
    
class sinsin(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=sincos() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: The function value at x
    """
    def __init__(self,f1=1,f2=0.1):
        self.input_dim=1
        self.bounds={'x':(-1,2)}
        self.f1=f1
        self.f2=f2
        self.fmin=11
        self.min=0
        self.ismax=1
        self.name='sinsin'
    def func(self,x):
        x=np.asarray(x)
        fval=np.sin(self.f1*x)+np.sin(self.f2*x)
        return fval*self.ismax

class sin(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=sincos() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: The function value at x
    """
    def __init__(self):
        self.input_dim=1
        self.bounds={'x':(-1,2)}

        self.fmin=11
        self.min=0
        self.ismax=1
        self.name='sin'
    def func(self,x):
        x=np.asarray(x)
        x=x*3-1
        fval=x*np.sin(x)
        return fval*self.ismax

class fourier(functions):
	"""
	Description: Function for BO method evaluation
	Useage: must be initiated with a f=fourier() call. It can then be
	evaluated with f.func(x) for the input x.
	Output: The function value at x
	"""
	def __init__(self,sd=None):
		self.input_dim = 1		
		self.sd=self.findSdev()
		self.min = 4.795 		## approx
		self.fmin = -9.5083483926941064 			## approx
		self.bounds = {'x':(0,10)}
		self.name='sincos'
		self.ismax=-1
	def func(self,X):
		X=np.asarray(X)*10
		X = X.reshape((len(X),1))
		n = X.shape[0]
		fval = X*np.sin(X)+X*np.cos(2*X)
		fval=self.ismax*fval.reshape(n,1)       
		return fval
        
        
class branin(functions):
	"""
	Description: Function for BO method evaluation
	Useage: must be initiated with a f=branin() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: The function value at x
    """
	def __init__(self,sd=None):
		self.input_dim=2
		self.bounds=OrderedDict([('x1',(-5,10)),('x2',(0,15))])
		self.fmin=0.397887
		self.min=[9.424,2.475]
		self.ismax=-1
		self.name='branin'

		self.sd=0
		self.sd=self.findSdev()
	def func(self,X):
		X=np.asarray(X)

		if len(X.shape)==1:
			x1=X[0]/max_bound_size
			x2=X[1]/max_bound_size
		else:
			x1=X[:,0]/max_bound_size
			x2=X[:,1]/max_bound_size
		x1=(x1*15-5)
		x2=(x2*15)

		a=1
		b=5.1/(4*np.power(np.pi,2))
		c=5/np.pi
		r=6
		s=10
		t=1/(8*np.pi)
		fx=a*np.power(x2-b*x1*x1+c*x1-r,2)+s*(1-t)*np.cos(x1)+s    
        
		n=X.shape[0]
		fval=fx*self.ismax
		return fval
        
    
class forrester(functions):
	"""
	Description: Function for BO method evaluation
	Useage: must be initiated with a f=forrester() call. It can then be
	evaluated with f.func(x) for the input x.
	Output: The function value at x
	"""
	def __init__(self):
		self.input_dim = 1		
		self.min = 0.78 		## approx
		self.fmin = -6.03 			## approx
		self.bounds = {'x':(0,1)}
		self.ismax=-1
		self.name='forrester'
		self.sd=0
		self.sd=self.findSdev()
      
	def func(self,x):
		x=np.asarray(x)
		fval = ((6*x -2)**2)*np.sin(12*x-4)
		noise = np.random.normal(0,0.1*self.sd,1).reshape(1,1)
		return fval*self.ismax+np.ravel(noise)

  
class rosenbrock(functions):
	"""
	Description: Function for BO method evaluation
	Useage: must be initiated with a f=rosenbrock() call. It can then be
	evaluated with f.func(x) for the input x.
	Output: fval, the function value at x
	"""
	def __init__(self,bounds=None):
		self.input_dim = 2
		if bounds == None: self.bounds = OrderedDict([('x1',(-0.5,3)),('x2',(-1.5,2))])
		else: self.bounds = bounds
		self.min = [(0, 0)]
		self.fmin = 0
		self.sd=0
		self.ismax=-1
		self.name = 'Rosenbrock'
		self.sd=self.findSdev()

	def func(self,X):
		Xc=np.copy(np.asarray(X))//max_bound_size
		n=1
		if len(X.shape)==1:# one observation
			x1=Xc[0]*3.5-0.5
			x2=Xc[1]*3.5-1.5
            
		else:# multiple observations
			x1=(Xc[:,0]*3.5-0.5)
			x2=(Xc[:,1]*3.5-1.5)
			n=X.shape[0]

		fx = 100*(x2-x1**2)**2 + (x1-1)**2
		fval=fx*self.ismax
		return -fval
    


class beale(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=beale() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict({'x1':(-1,1),'x2':(-1,1)})
        else: self.bounds = bounds
        self.min = [(3, 0.5)]
        self.fmin = 0
        self.ismax=-1
        self.name = 'Beale'
        self.sd=0
        self.sd=self.findSdev()


    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        if len(Xc.shape)==1:
            x1=Xc[0]*2-1
            x2=X[1]*2-1
        else:
            x1=Xc[:,0]*2-1
            x2=Xc[:,1]*2-1
        fval = (1.5-x1+x1*x2)**2+(2.25-x1+x1*x2**2)**2+(2.625-x1+x1*x2**3)**2
        fval=self.ismax*fval
        return fval 
			


class dropwave(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=dropwave() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(-5.12,5.12)),('x2',(-5.12,5.12))])
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = -1
        self.ismax=1
        self.name = 'dropwave'
        self.sd=0
        self.sd=self.findSdev()
		
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        n=1
        if len(Xc.shape)==1:
            x1=Xc[0]*10.24-5.12
            x2=Xc[1]*10.24-5.12
        else:
            x1=Xc[:,0]*10.24-5.12
            x2=Xc[:,1]*10.24-5.12
        fval = - (1+np.cos(12*np.sqrt(x1**2+x2**2))) / (0.5*(x1**2+x2**2)+2) 
        fval=self.ismax*fval
        return -fval


class cosines(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=cosines() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(0,1)),('x2',(0,1))])
        else: self.bounds = bounds
        self.min = [(0.31426205,  0.30249864)]
        self.fmin = -1.59622468
        self.ismax=1
        self.sd=self.findSdev()
        self.name = 'Cosines'

    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        
        u = 1.6*x1-0.5
        v = 1.6*x2-0.5
        fval = 1-(u**2 + v**2 - 0.3*np.cos(3*np.pi*u) - 0.3*np.cos(3*np.pi*v) )
        fval = self.ismax*fval
        return fval
            
            
            
class goldstein(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=goldstein() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-2,2),'x2':(-2,2)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 3
        self.ismax=-1

        self.name = 'Goldstein'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*4-2
            x2=X[1]*4-2
        else:
            x1=X[:,0]*4-2
            x2=X[:,1]*4-2
        x1=x1//max_bound_size
        x2=x2//max_bound_size
   
        fact1a = (x1 + x2 + 1)**2
        fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
        fact1 = 1 + fact1a*fact1b
        fact2a = (2*x1 - 3*x2)**2
        fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
        fact2 = 30 + fact2a*fact2b
        fval = fact1*fact2
        fval = self.ismax*fval
        return fval

class bukin6(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=bukin6() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-15,-5),'x2':(-3,3)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=-1

        self.name = 'Bukin N. 6'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*20-5
            x2=X[1]*6-3
        else:
            x1=X[:,0]*20-5
            x2=X[:,1]*6-3
        fval = 100*np.sqrt(np.abs(x2-0.01*np.square(x1)))+0.01*np.abs(x1+10)
        fval = self.ismax*fval
        return fval

class crossInTray(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=crossInTray() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-10,10),'x2':(-10,10)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = -2.06261
        self.ismax=-1

        self.name = 'Cross-in-tray'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*20-10
            x2=X[1]*20-10
        else:
            x1=X[:,0]*20-10
            x2=X[:,1]*20-10
        fval = -0.0001*np.power(np.abs(np.sin(x1)*np.sin(x2)*np.exp(100-np.sqrt(np.square(x1)+np.square(x2))/math.pi))+1,0.1)
        fval = self.ismax*fval
        return fval

class holdertable(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=holdertable() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-10,10),'x2':(-10,10)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = -19.2985
        self.ismax=-1

        self.name = 'Holder Table'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*20-10
            x2=X[1]*20-10
        else:
            x1=X[:,0]*20-10
            x2=X[:,1]*20-10
        fval = -np.abs(np.sin(x1)*np.cos(x2)*np.exp(1-np.sqrt(np.square(x1)+np.square(x2))/math.pi))
        fval = self.ismax*fval
        return fval

class levyN13(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=levyN13() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-10,10),'x2':(-10,10)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=-1

        self.name = 'Levy N. 13'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*20-10
            x2=X[1]*20-10
        else:
            x1=X[:,0]*20-10
            x2=X[:,1]*20-10
        fval = np.square(np.sin(3*math.pi*x1))+np.square(x1-1)*(1+np.square(np.sin(3*math.pi*x2)))+np.square(x2-1)*(1+np.square(np.sin(2*math.pi*x2)))
        fval = self.ismax*fval
        return fval

class SchafferN2(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=SchafferN2() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-100,100),'x2':(-100,100)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=-1

        self.name = 'Schaffer N. 2'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*200-100
            x2=X[1]*200-100
        else:
            x1=X[:,0]*200-100
            x2=X[:,1]*200-100
        fval = 0.5+(np.square(np.sin(np.square(x1)+np.square(x2)))-0.5)/np.square(1+0.001*(np.square(x1)+np.square(x2)))
        fval = self.ismax*fval
        return fval

class SchafferN4(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=SchafferN4() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-100,100),'x2':(-100,100)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=1

        self.name = 'Schaffer N. 4'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*200-100
            x2=X[1]*200-100
        else:
            x1=X[:,0]*200-100
            x2=X[:,1]*200-100
        fval = 0.5+(np.cos(np.sin(np.square(x1)+np.square(x2)))-0.5)/np.square(1+0.001*(np.square(x1)+np.square(x2)))
        fval = self.ismax*fval
        return fval

class Bohachevsky(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=Bohachevsky() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-100,100),'x2':(-100,100)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=-1

        self.name = 'Bohachevsky'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*200-100
            x2=X[1]*200-100
        else:
            x1=X[:,0]*200-100
            x2=X[:,1]*200-100
        fval = np.square(x1)+2*np.square(x2)-0.3*np.cos(3*math.pi*x1)-0.4*np.cos(4*math.pi*x2)+0.7
        fval = self.ismax*fval
        return fval


class Rastrigin(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=Rastrigin() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self, dim=2,bounds=None,sd=None):
        self.input_dim = dim
        if bounds == None: self.bounds = {'x1':(-10,10),'x2':(-10,10)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=-1

        self.name = 'Rastrigin'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        x=np.copy(X)
        fval=10*self.input_dim
        if len(X.shape)==1:
            for d in range(0,self.input_dim):
                x[d]=X[0]*10.24-5.12
                fval=fval+np.square(x[d])-10*np.cos(10*math.pi*x[d])
        else:
            for d in range(0,self.input_dim):
                x[:,d]=X[:,d]*10.24-5.12
                fval=fval+np.square(x[:,d])-10*np.cos(10*math.pi*x[:,d])
        fval = self.ismax*fval
        return fval

class Sphere(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=Sphere() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self, dim=4,bounds=None,sd=None):
        self.input_dim = dim
        if bounds == None:
            self.bounds =[(-5.12,5.12)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name = 'Sphere'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        x=np.copy(X)
        fval=10*self.input_dim
        if len(X.shape)==1:
            for d in range(0,self.input_dim):
                x[d]=X[0]*10.24-5.12
                fval=fval+np.square(x[d])
        else:
            for d in range(0,self.input_dim):
                x[:,d]=X[:,d]*10.24-5.12
                fval=fval+np.square(x[:,d])
        fval = self.ismax*fval
        return fval


class Matyas(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=Matyas() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-10,10),'x2':(-10,10)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=-1

        self.name = 'Matyas'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*20-10
            x2=X[1]*20-10
        else:
            x1=X[:,0]*20-10
            x2=X[:,1]*20-10
        fval = 0.26*(np.square(x1)+np.square(x2))-0.48*x1*x2
        fval = self.ismax*fval
        return fval

class Mccormick(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=Mccormick() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-10,10),'x2':(-10,10)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=-1

        self.name = 'Mc Cormick'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*5.5-1.5
            x2=X[1]*7-3
        else:
            x1=X[:,0]*5.5-1.5
            x2=X[:,1]*7-3
        fval = np.sin(x1+x2)+np.square(x1+x2)-1.5*x1+2.5*x2+1
        fval = self.ismax*fval
        return fval
    
class threehumpcamel(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=threehumpcamel() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-5,5),'x2':(-5,5)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=-1

        self.name = 'Three-hump camel'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]*10-5
            x2=X[1]*10-5
        else:
            x1=X[:,0]*10-5
            x2=X[:,1]*10-5
        fval = 2*np.square(x1)-1.05*np.power(x1,4)+np.power(x1,6)/6+x1*x2+np.square(x2)
        fval = self.ismax*fval
        return fval

class Dejong(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=Dejong() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-65.536,65.536),'x2':(-65.536,65.536)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 0
        self.ismax=-1

        self.name = 'De Jong N. 5'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        a=np.array([[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],[-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]])
        if len(X.shape)==1:
            x1=X[0]*131.072-65.536
            x2=X[1]*131.072-65.536
        else:
            x1=X[:,0]*131.072-65.536
            x2=X[:,1]*131.072-65.536
        fval =0.002
        for i in range(0,25):
            fval=fval+1/(i+1+np.power(x1-a[0,i],6)+np.power(x2-a[1,i],6))
        fval=1/fval
        fval = self.ismax*fval
        return fval

class Eason(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=Eason() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-100,100),'x2':(-100,100)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = -1
        self.ismax=-1

        self.name = 'Eason'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        if len(X.shape)==1:
            x1=X[0]*200-100
            x2=X[1]*200-100
        else:
            x1=X[:,0]*200-100
            x2=X[:,1]*200-100
        fval=-np.cos(x1)*np.cos(x2)*np.exp(-np.square(x1-math.pi)-np.square(x2-math.pi))
        fval = self.ismax*fval
        return fval


class Colville(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=Colville() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 4
        if bounds == None: self.bounds = {'x1':(-10,10),'x2':(-10,10),'x3':(-10,10),'x4':(-10,10)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = -1
        self.ismax=-1

        self.name = 'Colville'


    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        if len(X.shape)==1:
            x1=X[0]*20-10
            x2=X[1]*20-10
            x3=X[2]*20-10
            x4=X[3]*20-10
        else:
            x1=X[:,0]*20-10
            x2=X[:,1]*20-10
            x3=X[:,2]*20-10
            x4=X[:,3]*20-10            
        fval=100*(np.square(x1)-np.square(x2))+np.square(x1-1)+np.square(x3-1)+90*np.square(np.square(x3)-x4)+10.1*(np.square(x2-1)+np.square(x4-1))+19.8*(x2-1)*(x4-1)
        fval = self.ismax*fval
        return fval
#class langermann(functions):
#    """
#    Description: Function for BO method evaluation
#    Useage: must be initiated with a f=langermann() call. It can then be
#    evaluated with f.func(x) for the input x.
#    Output: fval, the function value at x
#    """
#    def __init__(self,bounds=None,sd=None):
#        self.input_dim = 2
#        if bounds == None: self.bounds = {'x1':(-10,10),'x2':(-10,10)}
#        else: self.bounds = bounds
#        self.min = [(0,-1)]
#        self.fmin = -19.2985
#        self.ismax=-1
#
#        self.name = 'Langermann'
#
#
#    def func(self,X):
#        X=np.asarray(X)
#        X = reshape(X,self.input_dim)
#
#        if len(X.shape)==1:
#            x1=X[0]*10-0
#            x2=X[1]*10-0
#        else:
#            x1=X[:,0]*10-0
#            x2=X[:,1]*10-0
#        fval=0
#        c=[1,2,5,2,3]
#        A=np.array([3,5,2,1,7],[5,2,1,4,9]).T
#        fval = -np.abs(np.sin(x1)*np.cos(x2)*np.exp(1-np.sqrt(np.square(x1)+np.square(x2))/math.pi))
#        for ci in c:
#            fval=fval+ci*np.exp((-1/math.pi)*np.sum())*np.cos()
#        fval = self.ismax*fval
#        return fval

class sixhumpcamel(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=sixhumpcamel() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(-2,2)),('x2',(-1,1))])
        else: self.bounds = bounds
        self.min = [(0.0898,-0.7126),(-0.0898,0.7126)]
        self.fmin = -1.0316
        self.ismax=-1
		 #if sd==None: self.sd = 0
		 #else: self.sd=sd      
        self.name = 'Six-hump camel'
        self.sd=0
        self.sd=self.findSdev()
		
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        n=1
        if len(Xc.shape)==1:
            x1=Xc[0]*4-2
            x2=Xc[1]*2-1
        else:
            x1=Xc[:,0]*4-2
            x2=Xc[:,1]*2-1
        term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
        term2 = x1*x2
        term3 = (-4+4*x2**2) * x2**2
        fval = term1 + term2 + term3
        fval = self.ismax*fval       
        return -fval



class mccormick(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=mccormick() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = [(-1.5,4),(-3,4)]
        else: self.bounds = bounds
        self.min = [(-0.54719,-1.54719)]
        self.fmin = -1.9133
        self.ismax=-1
        self.name = 'Mccormick'

    def func(self,X):
        X = reshape(X,self.input_dim)

        x1=X[:,0]*5.5-1.5   
        x2=X[:,1]+7-3
 
      
        term1 = np.sin(x1 + x2)
        term2 = (x1 - x2)**2
        term3 = -1.5*x1
        term4 = 2.5*x2
        fval = term1 + term2 + term3 + term4 + 1
        fval = self.ismax*fval 
        return self.ismax*fval


class powers(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=powers() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0,0)]
        self.fmin = 0
		 #if sd==None: self.sd = 0
		 #else: self.sd=sd
        self.sd=self.findSdev()
        self.name = 'Sum of Powers'

    def func(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = (x[:,0]*2-1)/max_bound_size
            x2 = (x[:,1]*2-1)/max_bound_size
            fval = abs(x1)**2 + abs(x2)**3
            fval = fval.reshape(n,1)
            return fval

class eggholder(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=eggholder() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None):
        self.input_dim = 2
        self.bounds = [(-512,512),(-512,512)]
        self.sd=0
        self.min = [(512,404.2319)]
        self.fmin = -959.6407
        self.ismax=-1

        self.name = 'egg-holder'
        self.sd=self.findSdev()
        

    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size


        x1=Xc[:,0]*1024-512
        x2=Xc[:,1]*1024-512
        fval = -(x2+47) * np.sin(np.sqrt(abs(x2+x1/2+47))) + -x1 * np.sin(np.sqrt(abs(x1-(x2+47))))
        fval = self.ismax*fval
        return -fval


class alpine1(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=alpine1() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """

    def __init__(self,input_dim, bounds=None, sd=None):
        if bounds == None: 
            self.bounds = bounds  =[(-10,10)]*input_dim
        else: 
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.fmin = 0
        self.input_dim = input_dim
        if sd==None: 
            self.sd = 0
        else: 
            self.sd=sd
            
        self.ismax=-1
        self.name='alpine1'
        self.sd=0
        self.sd=self.findSdev()
		
    def func(self,X):
        Xc = (np.copy(reshape(X,self.input_dim))/max_bound_size)*20-10

        temp=(Xc*np.sin(Xc) + 0.1*Xc)
        if len(temp.shape)<=1:
            fval=np.sum(temp)
        else:
            fval = np.sum(temp,axis=1)
        fval = self.ismax*fval
   
        return fval


class alpine2(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=alpine2() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,input_dim, bounds=None, sd=None):
        if bounds == None: 
            self.bounds = bounds  =[(1,10)]*input_dim
        else: 
            self.bounds = bounds
        self.min = [(7.917)]*input_dim
        self.fmin = -2.808**input_dim
        self.ismax=-1
        self.input_dim = input_dim
#        if sd==None: 
#            self.sd = 0
#        else: 
#            self.sd=sd
        self.name='Alpine2'
        self.sd=0
        self.sd=self.findSdev()	
		
    def internal_func(self,X):
        fval = np.cumprod(np.sqrt(X))[self.input_dim-1]*np.cumprod(np.sin(X))[self.input_dim-1]  
        return fval

    def func(self,X):
        Xc=np.copy(X)
        Xc = (reshape(Xc,self.input_dim)/max_bound_size)*10
        fval=[self.ismax*self.internal_func(val) for idx, val in enumerate(Xc)]
        fval=np.asarray(fval) 
        return fval

class gSobol:
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=gSobol() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,a,bounds=None,sd=None):
        self.a = a
        self.input_dim = len(self.a)

        if bounds == None: 
            self.bounds =[(-4,6)]*self.input_dim
        else: 
            self.bounds = bounds

        if not (self.a>0).all(): return 'Wrong vector of coefficients, they all should be positive'
        self.S_coef = (1/(3*((1+self.a)**2))) / (np.prod(1+1/(3*((1+self.a)**2)))-1)
        if sd==None: self.sd = 0
        else: self.sd=sd

        self.ismax=-1
        self.fmin=0# in correct
        self.name='gSobol'

    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]*10-4
        aux = (abs(4*X-2)+np.ones(n).reshape(n,1)*self.a)/(1+np.ones(n).reshape(n,1)*self.a)
        fval =  np.cumprod(aux,axis=1)[:,self.input_dim-1]
        fval = self.ismax*fval
        return self.ismax*fval

#####
class ackley(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=ackley() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim

        if bounds == None: 
            self.bounds =[(-32.768,32.768)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='ackley'
        self.sd=0
        self.sd=self.findSdev()
        
    def func(self,X):
        n=1
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        Xc = Xc*65.536-32.768
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((Xc**2).sum(1)/self.input_dim))-np.exp(np.cos(2*np.pi*Xc).sum(1)/self.input_dim))
        fval = self.ismax*fval
      
        return fval


#####
class hartmann_6d(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=hartmann_6d() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6

        if bounds == None: 
            self.bounds =[(0,1)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.ismax=-1
        self.name='hartmann_6d'
        self.sd=0
        self.sd=self.findSdev()
        
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        n = Xc.shape[0]

        
        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A=np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
			[2329, 4135, 8307, 3736, 1004, 9991],
			[2348, 1451, 3522, 2883, 3047, 6650],
			[4047, 8828, 8732, 5743, 1091, 381]]



        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;

        fval  =np.zeros((n,1))  
        for idx in range(n):
            outer = 0;
            for ii in range(4):
                inner = 0;
                for jj in range(6):
                    xj = Xc[idx,jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
				
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            fval[idx] = -(2.58 + outer) / 1.94;
            noise=0
      
        if n==1:
            return self.ismax*(fval[0][0])+noise
        else:
            return self.ismax*(fval)+noise
        
        
#####
class hartmann_4d:
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=hartmann_4d() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4

        if bounds == None: 
            self.bounds =[(0,1)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.ismax=-1
        self.name='hartmann_4d'
        
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        n = Xc.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A=np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
			[2329, 4135, 8307, 3736, 1004, 9991],
			[2348, 1451, 3522, 2883, 3047, 6650],
			[4047, 8828, 8732, 5743, 1091, 381]]



        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;
        

        fval  =np.zeros((n,1))        
        for idx in range(n):
            X_idx=Xc[idx,:]
            outer = 0;
            for ii in range(4):
                inner = 0;
                for jj in range(4):
                    xj = X_idx[jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = (1.1 - outer) / 0.839;
        noise=0
        if n==1:
            return self.ismax*(fval[0][0])+noise
        else:
            return self.ismax*(fval)+noise
            
            
            
class hartmann_3d(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=hartmann_4d() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,  bounds=None):
        self.input_dim = 3
        self.sd=0
        if bounds == None: 
            self.bounds =[(0,1)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.86278
        self.ismax=-1
        self.name='hartmann_3d'
        self.sd=self.findSdev()
        
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        n = Xc.shape[0]

        
        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[3.0, 10, 30],
             [0.1, 10, 35],
             [3.0, 10, 30],
             [0.1, 10, 35]]
        A=np.asarray(A)
        P = [[3689, 1170, 2673],
			[4699, 4387, 7470],
			[1091, 8732, 5547],
			[381, 5743, 8828]]



        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;

        fval  =np.zeros((n,1))  
        for idx in range(n):
            outer = 0;
            for ii in range(4):
                inner = 0;
                for jj in range(3):
                    xj = Xc[idx,jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
				
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            fval[idx] = -outer;
        noise=0
        if n==1:
            return self.ismax*(fval[0][0])+noise
        else:
            return self.ismax*(fval)+noise
    
class doubleGaussian(functions):
    """
    Description: The two peak Gaussian mixture function described in the paper. 
    Used for BO method evaluation.
    Useage: must be initiated with a f=doubleGaussian() call. It can then be
    evaluated with f.func(x) for the input x. Optionally a dimension parameter,
    d, can be added to the call, f=doubleGaussian(dim=d).
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None, dim=2):
        self.input_dim=dim
        self.sd=0
        if bounds == None:
            self.bounds =[(0,1)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fmin=-1
        self.ismax=-1
        self.name="doubleGaussian"
        self.sd=self.findSdev()
        self.functionMin,self.functionMax=(0,0)
        self.functionMin,self.functionMax=self.findExtrema()
    def func(self,X):
        X = reshape(X,self.input_dim)/max_bound_size
        n = X.shape[0]
        y1=multivariate_normal.pdf(X,mean=0.7*np.ones(self.input_dim),cov=0.01*np.eye(self.input_dim))
        y2=multivariate_normal.pdf(X,mean=0.1*np.ones(self.input_dim),cov=0.001*np.eye(self.input_dim))
        fval=y1+y2
        return fval
    
    
class gaussianMixtureExt(functions):
    """
    Description: The two peak Gaussian mixture function described in the paper. 
    Used for BO method evaluation.
    Useage: must be initiated with a f=doubleGaussian() call. It can then be
    evaluated with f.func(x) for the input x. Optionally a dimension parameter,
    d, can be added to the call, f=doubleGaussian(dim=d).
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None, dim=2):
        self.input_dim=dim
        self.sd=0
        if bounds == None:
            self.bounds =[(0,1)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fmin=-1
        self.ismax=-1
        self.name="doubleGaussian"
        self.sd=self.findSdev()
        self.functionMin,self.functionMax=(0,0)
        self.functionMin,self.functionMax=self.findExtrema()
    def func(self,X):
        X = reshape(X,self.input_dim)/max_bound_size
        n = X.shape[0]
        y1=1.75*multivariate_normal.pdf(X,mean=0.75*np.ones(self.input_dim),cov=0.1*np.eye(self.input_dim)/max_bound_size)
        y2=0.75*multivariate_normal.pdf(X,mean=0.5*np.ones(self.input_dim),cov=0.05*np.eye(self.input_dim)/max_bound_size)
        fval=y1+y2
        return fval
    
class schwefel(functions):
    """
    Description: Schwefel function described in the paper. 
    Used for BO method evaluation.
    Useage: must be initiated with a f=schwefel() call. It can then be
    evaluated with f.func(x) for the input x. Optionally a dimension parameter,
    d, can be added to the call, f=schwefel(dim=d)
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None, dim=3):
        self.input_dim=dim
        self.sd=0
        if bounds == None:
            self.bounds =[(0,1)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fmin=-420.9686
        self.ismax=-1
        self.name="schwefel"
        self.sd=self.findSdev()
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        Xc=1000*Xc-500
        if self.input_dim>1:
            fval=418.9829*self.input_dim-np.sum(Xc*np.sin(np.sqrt(np.abs(Xc))),axis=1)
        else:
            fval=418.9829-Xc*np.sin(np.sqrt(np.abs(Xc)))
        return fval

class shubert(functions):
    """
    Description: Shubert function described in the paper. 
    Used for BO method evaluation.
    Useage: must be initiated with a f=schwefel() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None):
        self.input_dim=2
        #self.sd=0
        if bounds == None:
            self.bounds =[(0,1)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fmin=-186.7309
        self.ismax=-1
        self.name="shubert"
        self.sd=self.findSdev()
        self.functionMin,self.functionMax=(0,0)
        self.functionMin,self.functionMax=self.findExtrema()
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        Xc=10.24*Xc-5.12
        n = X.shape[0]
        y1=0
        y2=0
        for i in range(1,5):
            y1+=i*np.cos((i+1)*Xc[:,0]+i)
            y2+=i*np.cos((i+1)*Xc[:,1]+i)
        fval=y1+y2
        return -fval  

class franke(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=franke() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None):
        self.input_dim=2
        #self.sd=0
        if bounds == None:
            self.bounds =[(0,1)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fmin=-186.7309
        self.ismax=-1
        self.name="franke"
        #self.sd=self.findSdev()
        self.functionMin,self.functionMax=(0,0)
        self.functionMin,self.functionMax=self.findExtrema()
    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        
        fval=0.75*np.exp(-np.square(9*X[:,0]-2)/4-np.square(9*X[:,1]-2)/4)
        fval+=0.75*np.exp(-np.square(9*X[:,0]+1)/49-(9*X[:,1]+1)/10)
        fval+=0.75*np.exp(-np.square(9*X[:,0]-7)/4-np.square(9*X[:,1]-3)/4)
        fval-=0.2*np.exp(-np.square(9*X[:,0]-4)-np.square(9*X[:,1]-7)) 
        return fval 

class griewank(functions):
    """
    Description: Function for BO method evaluation
    Useage: must be initiated with a f=griewank() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None, dim=3):
        self.input_dim=dim
        #self.sd=0
        if bounds == None:
            self.bounds =[(0,1)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fmin=-1
        self.ismax=-1
        self.name="griewank"
        #self.sd=self.findSdev()
        self.functionMin,self.functionMax=(0,0)
        self.functionMin,self.functionMax=self.findExtrema()
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        Xc=120*Xc-60
        n = X.shape[0]
        if self.input_dim>1:
            fval=np.sum(np.square(Xc)/4000,axis=1)-np.prod(np.cos(Xc/np.sqrt(np.arange(1,self.input_dim+1))),axis=1)+1
        else:
            fval=np.square(Xc)/4000-np.cos(Xc)+1
        return fval
    
class levy(functions):
    """
    Description: Shubert function described in the paper. 
    Used for BO method evaluation.
    Useage: must be initiated with a f=levy() call. It can then be
    evaluated with f.func(x) for the input x.
    Output: fval, the function value at x
    """
    def __init__(self,bounds=None, dim=3):
        self.input_dim=dim
        #self.sd=0
        if bounds == None:
            self.bounds =[(0,1)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fmin=-1
        self.ismax=-1
        self.name="levy"
        #self.sd=self.findSdev()
        self.functionMin,self.functionMax=(0,0)
        self.functionMin,self.functionMax=self.findExtrema()
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc = reshape(Xc,self.input_dim)
        Xc=Xc/max_bound_size
        Xc=20*Xc-10
        n = Xc.shape[0]
        if self.input_dim>1:
            fval=np.square(np.sin(math.pi*(1+(Xc[:,0]-1/4))))
            for d in range(0,self.input_dim-1):
                w=1+(Xc[:,d]-1)/4
                fval+=np.square(w-1)*(1+10*np.square(np.sin(math.pi*w+1)))
            w=1+(Xc[:,(self.input_dim-1)]-1)/4
            fval+=np.square(w-1)*(1+np.square(np.sin(2*math.pi*w)))
        else:
            fval=np.square(np.sin(math.pi*(1+(Xc[:]-1/4))))
            w=1+(Xc[:]-1)/4
            fval+=np.square(w-1)*(1+10*np.square(np.sin(math.pi*w+1)))
        return fval*self.ismax
    