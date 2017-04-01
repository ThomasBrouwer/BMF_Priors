'''
Measure convergence on the MovieLens 100K dataset, with the All Gaussian
model (multivariate posterior) and Wishart prior.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian_wishart import BMF_Gaussian_Gaussian_Wishart
from BMF_Priors.data.movielens.load_data import load_movielens_100K
from BMF_Priors.experiments.convergence.convergence_experiment import measure_convergence_time

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_movielens_100K()
model_class = BMF_Gaussian_Gaussian_Wishart
K = 20
settings = {
    'R': R, 
    'M': M, 
    'K': K, 
    'hyperparameters': { 'alpha':1., 'beta':1., 'mu0':0., 'beta0':1., 'v0':K, 'W0':1. }, 
    'init': 'random', 
    'iterations': 200,
}
fout_performances = './results/performances_gaussian_gaussian_wishart.txt'
fout_times = './results/times_gaussian_gaussian_wishart.txt'
repeats = 10
performances, times = measure_convergence_time(
    repeats, model_class, settings, fout_performances, fout_times)


''' Plot the times, and performance vs iterations. '''
plt.figure()
plt.title("Performance against average time")
plt.plot(times, performances)
plt.ylim(0,2000)

plt.figure()
plt.title("Performance against iteration")
plt.plot(performances)
plt.ylim(0,2000)
