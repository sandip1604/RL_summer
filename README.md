# RL_summer
summer rrsearch project with RL

## file and its discription

1. Gridenv.py - information of the enviroment ( check the Gridenv for each ALGs to make sure to have necessary hyper parameter tuned)

## ALG1 folder 

1. ALG1-3.py - has the Algorithm setup for probability for faulty dynamics set to 0.3

## ALG2 folder

1. Qvar-file.py - contains code to calculate the variance of return from the (state, action) pair using sampling method
2. variance.py - contains python class implementation that calulates the variance from the collected sample of rewards and transistion data.
3. Qvar_learning.py - contains alogrithm for Reinforcement learning along with learning variance like bellemn equation. That is learning method rather than sampling method. 
4. variance_learning.py - contains python class implementation that learns the variance rather than simply calculates it from the samples. 
5. opt_Qvar.py - contains code to optimise the hyperparamter "threshold" on variance to call the expert
