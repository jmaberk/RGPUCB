'''
Name: plot_results.py
Authors: Julian Berk and Vu Nguyen
Publication date:08/04/2019
Inputs: #Pickle files produced by bayesianOptimizationMaster.py
        #function_name: The name of the function used in the experiment as
        defined in functions.py
        #input_dim: The dimension of the function used
        #iteration_factor: This is multiplied with input_dim to determine the 
        number of iterations the experiments were run for
        #pickle_location: The location of the pickleStorage folder
Outputs: Plots the results contained in the input pickle files
Instructions:To suppress the results for an algorithm, comment the code bewteen
the "##ALGORITHM#NAME##" and "##ALGORITHM#NAME#END##" comments. The plot
settings eg. axis labels can be found at the end of the file, with the
exception of the figure size which is at the start
'''
###############################################################################
import sys
sys.path.insert(0,'../../')

import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from prada_bayes_opt import visualization
from prada_bayes_opt import auxiliary_functions
from prada_bayes_opt.utility import export_results


class plot(object):
    def __init__(self,function_name,input_dim,iteration_factor,pickle_location):
    
        sns.set(style="ticks")
        #change this to alter the figure size
        fig=plt.figure(figsize=(10, 7))
        
        
        ##############
        function_name=function_name
        D=input_dim
        y_optimal_value=0
        
        step=2
        mylinewidth=2
        std_scale=0.4
        
        T=iteration_factor*D+1
        BatchSz_GPyOpt=[1]*T
        BatchSz_GPyOpt[0]=D+1
        
        BatchSz_BigBound=[1]*T
        BatchSz_BigBound[0]=D+1
        
        x_axis=np.array(range(0,T+1))
        x_axis=x_axis[::step]
        
        # is minimization problem
        IsMin=1
        #IsMin=-1
        IsLog=0
        

######################################Thompson#################################     
        strFile=pickle_location+"\\{:s}_{:d}_Single_thompson.pickle".format(function_name,D)
        with open(strFile,'rb') as f:

            EI = pickle.load(f)
            
         
        myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(EI[0],BatchSz_BigBound,IsPradaBO=1,Y_optimal=1,step=step)
        ACSR_mean,ACSR_std=export_results.compute_average_cumulative_simple_regret(EI[0],BatchSz_GPyOpt,IsPradaBO=1,Y_optimal=y_optimal_value)
        print("EI ACSR {:.4f}({:.4f})".format(ACSR_mean,ACSR_std))
        myYbest=IsMin*np.asarray(myYbest)
        
        if IsLog==1:
            myYbest=np.log(myYbest)
            myStd=np.log(myStd)
        myStd=myStd*std_scale

        plt.errorbar(x_axis,-myYbest,yerr=myStd/np.sqrt(10),linewidth=mylinewidth,color='g',linestyle='-',marker='h', label='Thompson')
#####################################Thompson#END##############################
                
        
######################################GP-UCB###################################   
        strFile=pickle_location+"\\{:s}_{:d}_Single_ucb.pickle".format(function_name,D)
        with open(strFile,'rb') as f:

            EI = pickle.load(f)
            
         
        myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(EI[0],BatchSz_BigBound,IsPradaBO=1,Y_optimal=1,step=step)
        ACSR_mean,ACSR_std=export_results.compute_average_cumulative_simple_regret(EI[0],BatchSz_GPyOpt,IsPradaBO=1,Y_optimal=y_optimal_value)
        print("EI ACSR {:.4f}({:.4f})".format(ACSR_mean,ACSR_std))
        myYbest=IsMin*np.asarray(myYbest)
        
        if IsLog==1:
            myYbest=np.log(myYbest)
            myStd=np.log(myStd)
        myStd=myStd*std_scale

        plt.errorbar(x_axis,-myYbest,yerr=myStd/np.sqrt(10),linewidth=mylinewidth,color='b',linestyle='-',marker='h', label='GP-UCB')
######################################GP-UCB#END###############################   
        
######################################EI#######################################     
        strFile=pickle_location+"\\{:s}_{:d}_Single_ei.pickle".format(function_name,D)
        with open(strFile,'rb') as f:

            EI = pickle.load(f)
            
         
        myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(EI[0],BatchSz_BigBound,IsPradaBO=1,Y_optimal=1,step=step)
        ACSR_mean,ACSR_std=export_results.compute_average_cumulative_simple_regret(EI[0],BatchSz_GPyOpt,IsPradaBO=1,Y_optimal=y_optimal_value)
        print("EI ACSR {:.4f}({:.4f})".format(ACSR_mean,ACSR_std))
        myYbest=IsMin*np.asarray(myYbest)
        
        if IsLog==1:
            myYbest=np.log(myYbest)
            myStd=np.log(myStd)
        myStd=myStd*std_scale

        plt.errorbar(x_axis,-myYbest,yerr=myStd/np.sqrt(10),linewidth=mylinewidth,color='r',linestyle='-',marker='h', label='EI')
######################################EI#END##################################    

######################################RGP-UCB1###############################    
        strFile=pickle_location+"\\{:s}_{:d}_Single_rucb_exploit.pickle".format(function_name,D)
        with open(strFile,'rb') as f:

            EI = pickle.load(f)
            
         
        myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(EI[0],BatchSz_BigBound,IsPradaBO=1,Y_optimal=1,step=step)
        ACSR_mean,ACSR_std=export_results.compute_average_cumulative_simple_regret(EI[0],BatchSz_GPyOpt,IsPradaBO=1,Y_optimal=y_optimal_value)
        print("EI ACSR {:.4f}({:.4f})".format(ACSR_mean,ACSR_std))
        myYbest=IsMin*np.asarray(myYbest)
        
        if IsLog==1:
            myYbest=np.log(myYbest)
            myStd=np.log(myStd)
        myStd=myStd*std_scale

        plt.errorbar(x_axis,-myYbest,yerr=myStd/np.sqrt(10),linewidth=mylinewidth,color='black',linestyle='-',marker='h', label=r'RGP-UCB $\theta=0.5$')
######################################RGP-UCB1#END############################   

        
######################################RGP-UCB2###############################    
        strFile=pickle_location+"\\{:s}_{:d}_Single_rucb_balance.pickle".format(function_name,D)
        with open(strFile,'rb') as f:

            EI = pickle.load(f)
            
         
        myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(EI[0],BatchSz_BigBound,IsPradaBO=1,Y_optimal=1,step=step)
        ACSR_mean,ACSR_std=export_results.compute_average_cumulative_simple_regret(EI[0],BatchSz_GPyOpt,IsPradaBO=1,Y_optimal=y_optimal_value)
        print("EI ACSR {:.4f}({:.4f})".format(ACSR_mean,ACSR_std))
        myYbest=IsMin*np.asarray(myYbest)
        
        if IsLog==1:
            myYbest=np.log(myYbest)
            myStd=np.log(myStd)
        myStd=myStd*std_scale

        plt.errorbar(x_axis,-myYbest,yerr=myStd/np.sqrt(10),linewidth=mylinewidth,color='purple',linestyle='-',marker='h', label=r'RGP-UCB $\theta=1$')
######################################RGP-UCB2#END############################  
        
######################################RGP-UCB3###############################    
        strFile=pickle_location+"\\{:s}_{:d}_Single_rucb_explore.pickle".format(function_name,D)
        with open(strFile,'rb') as f:

            EI = pickle.load(f)
            
         
        myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(EI[0],BatchSz_BigBound,IsPradaBO=1,Y_optimal=1,step=step)
        ACSR_mean,ACSR_std=export_results.compute_average_cumulative_simple_regret(EI[0],BatchSz_GPyOpt,IsPradaBO=1,Y_optimal=y_optimal_value)
        print("EI ACSR {:.4f}({:.4f})".format(ACSR_mean,ACSR_std))
        myYbest=IsMin*np.asarray(myYbest)
        
        if IsLog==1:
            myYbest=np.log(myYbest)
            myStd=np.log(myStd)
        myStd=myStd*std_scale

        plt.errorbar(x_axis,-myYbest,yerr=myStd/np.sqrt(10),linewidth=mylinewidth,color='cyan',linestyle='-',marker='h', label=r'RGP-UCB $\theta=8$')
######################################RGP-UCB3#END############################ 
#        
        #The code below can be used to change the plot settings
        plt.ylabel('Best Found Value',fontdict={'size':22})
        plt.xlabel('Iteration',fontdict={'size':22})
        plt.legend(loc='top right',prop={'size':20})
        strTitle="{:s} D={:d}".format(function_name,D)
        plt.title(strTitle,fontdict={'size':28})
        
        plt.xticks(fontsize=18, rotation=0)
        plt.yticks(fontsize=18, rotation=0)
        
        strFile="plot/{:s}_{:d}_E3I.pdf".format(function_name,D)
        plt.savefig(strFile, bbox_inches='tight')
    

