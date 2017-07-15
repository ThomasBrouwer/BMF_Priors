'''
Measure the time taken for a couple of iterations.
'''

project_location = "/home/tab43/Documents/Projects/libraries" # "/Users/thomasbrouwer/Documents/Projects/libraries/" # 
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian import BMF_Gaussian_Gaussian
from BMF_Priors.data.movielens.load_data import load_movielens_1M

import time


''' Load data and set settings. '''
K = 10
hyperparameters = { 'alpha':1., 'beta':1., 'lamb':0.1 }
iterations = 10
init = 'random'

time0 = time.time()
R, M = load_movielens_1M()
time1 = time.time()
time_data = time1 - time0


''' Run without minpy (so numpy). '''
time2 = time.time()
BMF = BMF_Gaussian_Gaussian(R,M,K,hyperparameters) 
BMF.initialise(init)
BMF.run(iterations)
time3 = time.time()
time_model = time3 - time2


''' Print performances. '''
print "Time taken loading data: %s." % time_data
print "Time taken running model: %s." % time_model
