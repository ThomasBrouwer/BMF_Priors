'''
Measure the time taken for a couple of iterations.

On my laptop:
    MovieLens 100K:
        Time taken loading data: 2.32535505295.
        Time taken running model: 8.51722407341.
    MovieLens 1M:
        Time taken loading data: 102.270441055.
        Time taken running model: 81.7574019432.
On kiiara:
    MovieLens 100K:
        Time taken loading data: 3.51910114288.
        Time taken running model: 18.8004209995.
    MovieLens 1M:
        Time taken loading data: 101.879905939.
        Time taken running model: 119.432013988.
        
After making data loading more efficient:

On my laptop:
    MovieLens 100K:
        Time taken loading data: 0.0656440258026.
        Time taken running model: 8.53658103943.
    MovieLens 1M:
        Time taken loading data: 0.324141025543.
        Time taken running model: 83.3057899475.
On Darwin cluster:
    MovieLens 100K:
        Time taken loading data: 0.0200979709625.
        Time taken running model: 10.8945901394.
    MovieLens 1M:
        Time taken loading data: 0.193010091782.
        Time taken running model: 69.9377059937.
    
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/" # "/home/tab43/Documents/Projects/libraries" # 
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian import BMF_Gaussian_Gaussian
from BMF_Priors.data.movielens.load_data import load_processed_movielens_1M
from BMF_Priors.data.movielens.load_data import load_processed_movielens_100K

import time


''' Load data and set settings. '''
K = 10
hyperparameters = { 'alpha':1., 'beta':1., 'lamb':0.1 }
iterations = 10
init = 'random'

time0 = time.time()
R, M = load_processed_movielens_1M() # load_processed_movielens_100K() # 
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
