# -*- coding: utf-8 -*-
"""
Name: real_experiment_functions.py
Authors: Julian Berk and Vu Nguyen
Publication date:08/04/2019
Description: These classes run real-world experiments that can be used to test
our acquisition functions

###############################IMPORTANT#######################################
The classes here all have file paths that need to be set correctlt for them to
work. Please make sure you change all paths before using a class
"""

import numpy as np
from collections import OrderedDict
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVR
import math

import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
###############################################################################
'''
###IMPORTANT###
This variable must be consistant in all of the following files:
1) acquisition_functions.py
2) bayesian_optimization_function.py
3) function.py
4) real_experiment_functon.py
'''
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
    def plot(self):
        print("not implemented")
        
    
class SVR_function:
    '''
    SVR_function: function to run SVR for tetsing the our method. The default
    dataset is the Space GA but othe datasets can be used.
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        self.maxbounds = np.array([[0.1,1000*20],[0.000001,1*20],[0.00001,5*20]])
        if bounds == None: 
            self.bounds = OrderedDict([('C',(0.1,1000)),('epsilon',(0.000001,1)),('gamma',(0.00001,5))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='SVR on Space GA'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_SVR(self,X,X_train,y_train,X_test,y_test):
        Xc=np.copy(X)
        x1=Xc[0]*1000+0.1
        x2=Xc[1]+0.000001
        x3=Xc[2]*5+0.00001
        if x1<0.1:
            x1=0.1
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        # Fit regression model
        if(x1<=0):
            print("x1={}".format(x1))

        svr_model = SVR(kernel='rbf', C=x1, epsilon=x2,gamma=x3)
        svr_model.fit(X_train, y_train).predict(X_test)
        y_pred = svr_model.predict(X_test)
        
        squared_error=y_pred-y_test
        squared_error=np.mean(squared_error**2)
        
        RMSE=np.sqrt(squared_error)
        return RMSE
        
    def func(self,X):
        X=np.asarray(X)
        ##########################CHANGE PATH##################################    
#        Xdata, ydata = self.get_data(r"C:\Users\jmabe\OneDrive\Documents\PhD\Code\Bayesian\DDGP-UCB\real_experiment\space_ga_scale")
        Xdata, ydata = self.get_data(r"E:\OneDrive\Documents\PhD\Code\Bayesian\DDGP-UCB\real_experiment\space_ga_scale")
        nTrain=np.int(0.7*len(ydata))
        X_train, y_train = Xdata[:nTrain], ydata[:nTrain]
        X_test, y_test = Xdata[nTrain+1:], ydata[nTrain+1:]
        ###############################################################################
        # Generate sample data

        #y_train=np.reshape(y_train,(nTrain,-1))
        #y_test=np.reshape(y_test,(nTest,-1))
        ###############################################################################

        
        if len(X.shape)==1: # 1 data point
            RMSE=self.run_SVR(X,X_train,y_train,X_test,y_test)
        else:
            RMSE=np.zeros(X.shape[0])
            for i in range(0,np.shape(X)[0]):
                RMSE[i]=self.run_SVR(X[i],X_train,y_train,X_test,y_test)

        #print RMSE    
        return RMSE*self.ismax
        
class AlloyCooking_Profiling:
    '''
    Simulation for the cooking of an Aluminium Scandium alloy with two cooking stages
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(1*3600,3*3600)),('Time2',(1*3600,3*3600)),('Temp1',(200,300)),('Temp2',(300,400))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        if X.ndim==1:
            x1=X[0]
            x2=X[1]
            x3=X[2]
            x4=X[3]

        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.63;
        myxmatrix=0.0004056486;# dataset1
        myiSurfen=0.096;
        myfSurfen=1.58e-01;
        myRadsurfenchange=5e-09;
        
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
#        eng.addpath(r'C:\Users\jmabe\OneDrive\Documents\PhD\Code\Bayesian\DDGP-UCB\real_experiment\KWN_Heat_Treatment',nargout=0)
        eng.addpath(r'E:\OneDrive\Documents\PhD\Code\Bayesian\DDGP-UCB\real_experiment\KWN_Heat_Treatment',nargout=0)

        myCookTemp=matlab.double([x3,x4])
        myCookTime=matlab.double([x1,x2])
        strength,averad,phasefraction=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange,nargout=3)

        # minimize ave radius [0.5] and maximize phase fraction [0 2]
        temp_str=np.asarray(strength)
        temp_averad=np.asarray(averad)
        temp_phasefrac=np.asarray(phasefraction)
        return temp_str[0][1],temp_averad[0][1],temp_phasefrac[0][1]
        
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc=Xc
        if X.ndim==1:
            Xc[0]=Xc[0]*3*3600+3600
            Xc[1]=Xc[1]*3*3600+3600
            Xc[2]=Xc[2]*100+200
            Xc[3]=Xc[3]*100+200
            
        else:
            Xc[:,0]=Xc[:,0]*3*3600+3600
            Xc[:,1]=Xc[:,1]*3*3600+3600
            Xc[:,2]=Xc[:,2]*100+200
            Xc[:,3]=Xc[:,3]*100+200
        if len(Xc.shape)==1: # 1 data point
            Strength,AveRad,PhaseFraction=self.run_Profiling(Xc)

        else:

            temp=np.apply_along_axis( self.run_Profiling,1,Xc)
            Strength=temp[:,0]
            AveRad=temp[:,1]
            PhaseFraction=temp[:,2]



        #utility_score=-AveRad/13+PhaseFraction/2
        utility_score=Strength
        
        return utility_score    

class AlloyCooking_Profiling_3Steps:
    '''
    Simulation for the cooking of an Aluminium Scandium alloy with three cooking stages
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        self.maxbounds = np.array([[0,20],[0,20],[0,20],[0,20],[0,20],[0,20]])
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(1*3600,3*3600)),('Time2',(1*3600,3*3600)),('Time3',(1*3600,3*3600)),
                                       ('Temp1',(200,300)),('Temp2',(300,400)),('Temp3',(300,400))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        if X.ndim==1:
            x1=X[0]
            x2=X[1]
            x3=X[2]
            x4=X[3]
            x5=X[4]
            x6=X[5]
            
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.63;
        myxmatrix=0.0004056486;# dataset1
        myiSurfen=0.096;
        myfSurfen=1.58e-01;
        myRadsurfenchange=5e-09;
        
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        #eng.addpath(r'C:\Users\jmabe\OneDrive\Documents\PhD\Code\Bayesian\DDGP-UCB\real_experiment\KWN_Heat_Treatment',nargout=0)
        eng.addpath(r'E:\OneDrive\Documents\PhD\Code\Bayesian\DDGP-UCB\real_experiment\KWN_Heat_Treatment',nargout=0)
        myCookTemp=matlab.double([x4,x5,x6])
        myCookTime=matlab.double([x1,x2,x3])
        strength,averad,phasefraction=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange,nargout=3)

        # minimize ave radius [0.5] and maximize phase fraction [0 2]
        temp_str=np.asarray(strength)
        temp_averad=np.asarray(averad)
        temp_phasefrac=np.asarray(phasefraction)
        return temp_str[0][1],temp_averad[0][1],temp_phasefrac[0][1]
        
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        Xc=Xc
        if X.ndim==1:
            Xc[0]=Xc[0]*3*3600+3600
            Xc[1]=Xc[1]*3*3600+3600
            Xc[2]=Xc[2]*3*3600+3600
            Xc[3]=Xc[3]*100+200
            Xc[4]=Xc[4]*100+300
            Xc[5]=Xc[5]*100+300
            
        else:
            Xc[:,0]=Xc[:,0]*3*3600+3600
            Xc[:,1]=Xc[:,1]*3*3600+3600
            Xc[:,2]=Xc[:,2]*3*3600+3600
            Xc[:,3]=Xc[:,3]*100+200
            Xc[:,4]=Xc[:,4]*100+300
            Xc[:,5]=Xc[:,5]*100+300
        if len(Xc.shape)==1: # 1 data point
            Strength,AveRad,PhaseFraction=self.run_Profiling(Xc)

        else:

            temp=np.apply_along_axis( self.run_Profiling,1,Xc)
            Strength=temp[:,0]
            AveRad=temp[:,1]
            PhaseFraction=temp[:,2]


        utility_score=Strength
        
        return utility_score  
    
class Robot_BipedWalker:

    '''
    Robot Walker: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 8
        
        if bounds == None:  
            self.bounds = OrderedDict([('a1',(0,2)),('a2',(-1,1)),('a3',(-1,1)),('a4',(-6,-3)),
                                        ('a5',(-4,-3)),('a6',(2,4)),('a7',(3,5)),('a8',(-1,2))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='Robot_BipedWalker'
        
    
    def run_BipedWalker(self,X):
        #print X
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'E:\OneDrive\Documents\PhD\Code\Bayesian\DDGP-UCBB\real_experiment\WGCCM_three_link_walker_example\WGCCM_three_link_walker_example',nargout=0)
        #eng.addpath(r'C:\Users\jmabe\OneDrive\Documents\PhD\Code\Bayesian\DDGP-UCB\real_experiment\WGCCM_three_link_walker_example\WGCCM_three_link_walker_example',nargout=0)

        temp=matlab.double(X.tolist())


        hz_velocity=eng.walker_evaluation(temp[0])

        if math.isnan(hz_velocity) or math.isinf(hz_velocity):
            hz_velocity=0
        return hz_velocity
        
    def func(self,X):
        Xc=np.copy(np.asarray(X))
        if Xc.ndim>1:
            print("Xc[:,0].shape={}".format(Xc[:,0].shape))
            Xc[:,0]=Xc[:,0]*2
            Xc[:,1]=Xc[:,1]*2-1
            Xc[:,2]=Xc[:,2]*2-1
            Xc[:,3]=Xc[:,3]*3-6
            Xc[:,4]=Xc[:,4]-4
            Xc[:,5]=Xc[:,5]*4+2
            Xc[:,6]=Xc[:,6]*2+3
            Xc[:,7]=Xc[:,7]*2-1
        else:
            Xc[0]=Xc[0]*2
            Xc[1]=Xc[1]*2-1
            Xc[2]=Xc[2]*2-1
            Xc[3]=Xc[3]*3-6
            Xc[4]=Xc[4]-4
            Xc[5]=Xc[5]*4+2
            Xc[6]=Xc[6]*2+3
            Xc[7]=Xc[7]*2-1 
        if len(Xc.shape)==1: # 1 data point
            velocity=self.run_BipedWalker(Xc)
        else:

            velocity=np.apply_along_axis( self.run_BipedWalker,1,Xc)

        return velocity*self.ismax
	
class DeepLearning_CNN_MNIST:

    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 8
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('nb_filter',(10,50)),('nb_pool',(5,20)),('dropout1',(0.01,0.5)),('dense1',(64,200)),
                                        ('dropout2',(0.01,0.5)),('lr',(0.01,1)),('decay',(1e-8,1e-5)),('momentum',(0.5,1))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='DeepLearning_CNN_MNIST'
        
    
    def run_CNN_MNIST(self,X,X_train,Y_train,X_test,Y_test):
        #print X
        # Para: 512, dropout 0.2, 512, 0.2, 10
        
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten
        from keras.optimizers import SGD, Adam, RMSprop
        from keras.layers import Convolution2D, MaxPooling2D
        

        batch_size = 128 #var1
        nb_classes = 10
        nb_epoch = 1
        Xc=np.copy(np.asarray(X)) 
        if Xc.ndim>1:
            print("Xc[:,0].shape={}".format(Xc[:,0].shape))
            Xc[:,0]=Xc[:,0]*40+10
            Xc[:,1]=Xc[:,1]*15+5
            Xc[:,2]=Xc[:,2]*0.49+0.01
            Xc[:,3]=Xc[:,3]*136+64
            Xc[:,4]=Xc[:,4]*0.49+0.01
            Xc[:,5]=Xc[:,5]*0.09+0.01
            Xc[:,6]=Xc[:,6]*1e-5+1e-8
            Xc[:,7]=Xc[:,7]*0.5+0.5
        else:
            Xc[0]=Xc[0]*40+10
            Xc[1]=Xc[1]*15+5
            Xc[2]=Xc[2]*0.49+0.01
            Xc[3]=Xc[3]*136+64
            Xc[4]=Xc[4]*0.49+0.01
            Xc[5]=Xc[5]*0.09+0.01
            Xc[6]=Xc[6]*1e-5+1e-8
            Xc[7]=Xc[7]*0.5+0.5 
        # input image dimensions
        img_rows, img_cols = 28, 28
        # number of convolutional filters to use
        nb_filters = np.int(Xc[0]) #var1
        #nb_filters = 32
        # size of pooling area for max pooling
        nb_pool = np.int(Xc[1]) #var2
        #nb_pool = 2
        # convolution kernel size
        kernel_size = (3, 3)
        
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=(1, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        
        temp=np.int(Xc[2]*100)
        x3=temp*1.0/100
        model.add(Dropout(x3))#var3
        
        
        model.add(Flatten())
        
        temp=np.int(Xc[3]*100)
        x4=np.int(temp)
        print('x4={}'.format(x4))
        model.add(Dense(x4))#var4        
        model.add(Activation('relu'))
        
        temp=np.int(Xc[4]*100)
        x5=temp*1.0/100
        model.add(Dropout(x5))#var5
        
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        
        #model.summary()
        
        # learning rate, decay, momentum
        #sgd = SGD(lr=Xc[5], decay=Xc[6], momentum=Xc[7], nesterov=True)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=0, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        return score[1]
    def func(self,X):
        
        np.random.seed(1337)  # for reproducibility
        
        from keras.datasets import mnist

        from keras.utils import np_utils
    
        X=np.asarray(X)
        
        batch_size = 128
        nb_classes = 10
        nb_epoch = 1

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        #X_train = X_train.reshape(60000, 784)
        #X_test = X_test.reshape(10000, 784)
        #X_train = X_train.astype('float32')
        #X_test = X_test.astype('float32')
        #X_train /= 255
        #X_test /= 255
        #print(X_train.shape[0], 'train samples')
        #print(X_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        

        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_CNN_MNIST(X,X_train,Y_train,X_test,Y_test)
        else:
#            print(X.shape)
#            print(X_train.shape)
#            print(Y_train.shape)
#            print(X_test.shape)
#            print(Y_test.shape)
            Accuracy=np.apply_along_axis( self.run_CNN_MNIST,1,X,X_train,Y_train,X_test,Y_test)

        #print RMSE    
        return Accuracy*self.ismax  
        
        
         
class BayesNonMultilabelClassification:

    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        
        if bounds == None:  
            self.bounds = OrderedDict([('eta_xx',(0.0001,0.05)),('eta_yy',(0.000001,0.05)),('svi_rate',(0.000001,0.001)),('lambda',(30,60)),
                                        ('trunc',(0.000001,0.000005)),('alpha',(0.7,1.1))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='BNMC'
        
        #import matlab.engine
        #import matlab
        #eng = matlab.engine.start_matlab()
        
        """
        import scipy.io
        mydata=scipy.io.loadmat(r'P:\03.Research\05.BayesianOptimization\PradaBayesianOptimization\run_experiments\run_experiment_unbounded\BNMC\SceneData.mat')
        xxTrain=mydata['xxTrain']
        self.xxTrain=matlab.double(xxTrain.tolist())
        xxTest=mydata['xxTest']
        self.xxTest=matlab.double(xxTest.tolist())
        yyTrain=mydata['yyTrain']
        self.yyTrain=matlab.double(yyTrain.tolist())
        yyTest=mydata['yyTest']
        self.yyTest=matlab.double(yyTest.tolist())
        self.isloaded=1
        """
    def run_BNMC(self,X):
        #print X
        
#        eng.addpath(r'C:\Users\jmabe\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC\utilities',nargout=0)
#        eng.addpath(r'C:\Users\jmabe\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC\data',nargout=0)
#        eng.addpath(r'C:\Users\jmabe\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC',nargout=0)
        eng.addpath(r'E:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC\utilities',nargout=0)
        eng.addpath(r'E:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC\data',nargout=0)
        eng.addpath(r'E:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC',nargout=0)

        # convert variables
        temp=matlab.double(X.tolist())

        F1score=eng.BayesOpt_BNMC(temp[0])
        #F1score=eng.BayesOpt_BNMC(temp[0],self.xxTrain,self.xxTest,self.yyTrain,self.yyTest)
        #print F1score

        return F1score
        
    def func(self,X):
        #print X
        X=np.asarray(X)
            
        if len(X.shape)==1: # 1 data point
            F1score=self.run_BNMC(X)
        else:

            F1score=np.apply_along_axis( self.run_BNMC,1,X)

        return F1score*self.ismax 