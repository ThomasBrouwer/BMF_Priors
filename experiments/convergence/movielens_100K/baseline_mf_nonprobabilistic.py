'''
Measure convergence on the MovieLens 100K dataset, with the non-probabilistic MF baseline.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BMF_Priors.code.models.baseline_mf_nonprobabilistic import MF_Nonprobabilistic
from BMF_Priors.data.movielens.load_data import load_processed_movielens_100K
from BMF_Priors.experiments.convergence.convergence_experiment import measure_convergence_time

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_processed_movielens_100K()
model_class = MF_Nonprobabilistic
settings = {
    'R': R, 
    'M': M, 
    'K': 20, 
    'hyperparameters': { 'exponential_prior': 0.1 }, 
    'init': 'exponential', 
    'iterations': 200,
}
fout_performances = './results/performances_baseline_mf_nonprobabilistic.txt'
fout_times = './results/times_baseline_mf_nonprobabilistic.txt'
repeats = 10
performances, times = measure_convergence_time(
    repeats, model_class, settings, fout_performances, fout_times)


''' Plot the times, and performance vs iterations. '''
plt.figure()
plt.title("Performance against average time")
plt.plot(times, performances)
plt.ylim(0,10)

plt.figure()
plt.title("Performance against iteration")
plt.plot(performances)
plt.ylim(0,10)
