'''
Measure the time taken with, and without, the minpy library.
'''

project_location = "/home/tab43/Documents/Projects/libraries" # "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian import BMF_Gaussian_Gaussian
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer

import time


''' Load data and set settings. '''
R, M = load_gdsc_ic50_integer()
K = 10
hyperparameters = { 'alpha':1., 'beta':1., 'lamb':0.1 }
iterations = 20
init = 'random'


''' Run without minpy (so numpy). '''
time0_numpy = time.time()
BMF = BMF_Gaussian_Gaussian(R,M,K,hyperparameters) 
BMF.initialise(init)
BMF.run(iterations)
time1_numpy = time.time()
time_numpy = time1_numpy - time0_numpy

                      
''' Run with minpy. '''
import minpy.numpy as numpy
time0_minpy = time.time()
BMF = BMF_Gaussian_Gaussian(R,M,K,hyperparameters) 
BMF.initialise(init)
BMF.run(iterations)
time1_minpy = time.time()
time_minpy = time1_minpy - time0_minpy


''' Print performances. '''
print "Time taken by numpy: %s. \nTime taken by minpy: %s." % (time_numpy, time_minpy)
