'''
Measure the time taken with (or without) the minpy library.

In parameters.py, use either:
    import minpy.numpy as minpy (with minpy)
or
    import numpy as minpy (without minpy)
'''

project_location = "/home/tab43/Documents/Projects/libraries" # "/Users/thomasbrouwer/Documents/Projects/libraries/" # 
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
time0 = time.time()
BMF = BMF_Gaussian_Gaussian(R,M,K,hyperparameters) 
BMF.initialise(init)
BMF.run(iterations)
time1 = time.time()
time = time1 - time0


''' Print performances. '''
print "Time taken: %s." % time
