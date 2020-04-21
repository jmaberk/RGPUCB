'''
Name: example.py
Authors: Julian Berk
Publication date:15/01/2020

This is a simplified version of main.py intended to give a short example of 
RGP-UCB. Once the pickle directory is specified, it will run a shortened 
Bayesian optimisation algorithm on the alpine 2 benchmark function and create a
 plot summarising the results.
'''
###############################################################################
import sys
sys.path.insert(0,'../../')
import numpy as np
from prada_bayes_opt import auxiliary_functions
from prada_bayes_opt import functions
from prada_bayes_opt.bayesian_optimization_function import PradaBayOptFn
from prada_bayes_opt import real_experiment_function
from prada_bayes_opt.utility import export_results
import plot_results
import pickle
import random
import time
import matplotlib.pyplot as plt
#import pickle
import warnings
import itertools
warnings.filterwarnings("ignore")

'''
***********************************IMPORTANT***********************************
The pickle_location variable below must be changed to the appropriate directory
in your system for the code to work.
'''
pickle_location=r"E:\OneDrive\Documents\PhD\Code\Bayesian\RGP-UCB\pickleStorage"
#pickle_location=r"C:\yourPath\DDGP-UCB\pickleStorage"
###############################################################################
'''
Here the user can choose which functions to optimize. Just un-comment the 
desired functions and set the desired dimensions with the dim parameter
in supported functions
'''
###############################################################################
myfunction_list=[]

myfunction_list.append(functions.alpine2(input_dim=5))

###############################################################################
'''
Here the user can choose which acquisition functions will be used. To select
an acquisition function, un-comment the "acq_type_list.append(temp)" after its
name. If you do not have any pickle files for the method and function, you will
also need to comment out the relevent section in plot_results.py.
'''
###############################################################################
acq_type_list=[]

temp={}
temp['name']='ei'
acq_type_list.append(temp)

temp={}
temp['name']='ucb'
acq_type_list.append(temp)

temp={}
temp['name']='rucb_exploit'
acq_type_list.append(temp)

temp={}
temp['name']='rucb_explore'
acq_type_list.append(temp)

temp={}
temp['name']='rucb_balance'
acq_type_list.append(temp)

temp={}
temp['name']='thompson'
acq_type_list.append(temp)

mybatch_type_list={'Single'}
###############################################################################
kernel_type_list=[]                                                       

temp={}
temp='SE'
kernel_type_list.append(temp)

mybatch_type_list={'Single'}
###############################################################################
'''
#1 seed is used along with the experiment number as a seed to randomly generate
the initial points. Setting this as a constant will allow results to be
reproduced while making it random will let each set of runs use a different
set of initial points.
#2 num_initial_points controls the number of random sampled points each 
experiment will start with.
#3 max_iterations controls the number of iterations of Bayesian optimization
that will run on the function. This must be controlled with iteration_factor
for compatability with the print function.
#4 num_repeats controls the number of repeat experiments.
5# acq_params['optimize_gp'] If this is 1, then the lengthscale will be
determined by maximum likelihood every 15 samples. If any other value, no
lengthscale adjustement will be made
'''
###############################################################################
#seed=np.random.randint(1,100) #1
seed=1
print("Seed of {} used".format(seed))

for idx, (myfunction,acq_type,mybatch_type,) in enumerate(itertools.product(myfunction_list,acq_type_list,mybatch_type_list)):
    func=myfunction.func
    mybound=myfunction.bounds
    yoptimal=myfunction.fmin*myfunction.ismax
    
    acq_type['dim']=myfunction.input_dim 
    
    num_initial_points=3*myfunction.input_dim+1 #2
    
    iteration_factor=10 #3
    max_iterations=iteration_factor*myfunction.input_dim 
    
    num_repeats=5 #4
    
    GAP=[0]*num_repeats
    ybest=[0]*num_repeats
    Regret=[0]*num_repeats
    MyTime=[0]*num_repeats
    MyOptTime=[0]*num_repeats
    ystars=[0]*num_repeats

    func_params={}
    func_params['bounds']=myfunction.bounds
    func_params['f']=func

    acq_params={}
    acq_params["bb_function"]=myfunction
    acq_params['iteration_factor']=iteration_factor
    acq_params['acq_func']=acq_type
    acq_params['optimize_gp']=0 #5if 1 then maximum likelihood lenghscale selection will be used
#    acq_params['random_initial_bound']=1 #6 if 1 then the initial bound will be chosen at random
    acq_params['delta']=0.1
    acq_params['max_iterations']=max_iterations
    acq_params['num_initial_points']=num_initial_points 
    for kernel in kernel_type_list:    
        for ii in range(num_repeats):
            acq_params['iterations_num']=ii             
            gp_params = {'kernel_name':kernel, 'l':0.05,'noise_delta':0.001} # Kernel parameters for the square exponential kernel
            baysOpt=PradaBayOptFn(gp_params,func_params,acq_params,experiment_num=ii,seed=seed)
    
            ybest[ii],MyTime[ii]=auxiliary_functions.run_experiment(baysOpt,gp_params,
                                                    yoptimal,n_init=num_initial_points,NN=max_iterations)
                                                          
            MyOptTime[ii]=baysOpt.time_opt
            ystars[ii]=baysOpt.ystars
        Score={}
        Score["GAP"]=GAP
        Score["ybest"]=ybest
        Score["ystars"]=ystars
        Score["Regret"]=Regret
        Score["MyTime"]=MyTime
        Score["MyOptTime"]=MyOptTime
        export_results.print_result_ystars(baysOpt,myfunction,Score,mybatch_type,acq_type,acq_params,toolbox='PradaBO')

#Plots the results. Comment out to supress plots.
for idx, (myfunction) in enumerate(itertools.product(myfunction_list)):
    plot_results.plot(myfunction[0].name,myfunction[0].input_dim,iteration_factor,pickle_location)