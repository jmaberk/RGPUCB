# RGPUCB

This code runs the Bayesian optimisation with the search space defined with the distribution derived boundary (DDB) method. It also runs other similar methods to compare performance. The paper describing the algorithm has been accepted for publication in IJCAI-PRICAI2020, the 29th International Joint Conference on Artificial Intelligence and the 17th Pacific Rim International Conference on Artificial Intelligence.
NOTE: The code here is still being edited for clarity.


## System Requirements
This code was written for python 3.7. It may need to be modified to run on other versions of python. It also requires several standard python packages such as numpy, scipy, pickle, itertools, random, seaborn, matplotlib, sklearn, math, time, mpl_toolkits, and copy. 

The real world experiments require Matlab engine (https://au.mathworks.com/help/matlab/matlab-engine-for-python.html) and Keras (https://keras.io/). Keras itself requires TensorFlow, CNTK, or Theano. The rest of the code does not require these so it can be run without them by removing any imports of real_experiment_function.py.

We used the code on a windows 10 machine with Matlab R2015b and a Theano-based Keras.

## Previous work
Some code, including the real-world functions, are taken from previous works by Nguyen et al. These can be found here: https://github.com/ntienvu/ICDM2017_FBO and here: https://github.com/ntienvu/ICDM2016_B3O

## Example code
A quick example of the code is shown in example.py. This shows the optimization of the Hartmann 3D function.

## Usage
IMPORTANT: The pickle_location variable in bayesianOptimizationMaster.py must be changed to the location of your pickleStorage file for this code to run

The file bayesianOptimizationMaster.py controls the rest of the code It is set up to run a Hartmann 3D function but instructions running other functions and altering other parameters are given as comments in the code between rows of hash symbols.
