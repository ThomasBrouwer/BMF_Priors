'''
Measure sparsity experiment on the MovieLens 100K dataset, with the Gaussian + Laplace model.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_laplace import BMF_Gaussian_Laplace
from BMF_Priors.data.movielens.load_data import load_processed_movielens_100K
from BMF_Priors.experiments.sparsity.sparsity_experiment import sparsity_experiment

import matplotlib.pyplot as plt
import math


''' Run the experiment. '''
R, M = load_processed_movielens_100K()
model_class = BMF_Gaussian_Laplace
n_repeats = 10
stratify_rows = False
fractions_known = [0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
fractions_unknown = [1. - v for v in fractions_known]
settings = {
    'R': R, 
    'M': M, 
    'K': 5,
    'hyperparameters': { 'alpha':1., 'beta':1., 'eta':math.sqrt(10.) }, 
    'init': 'random', 
    'iterations': 250,
    'burn_in': 200,
    'thinning': 1,
}
fout = './results/performances_gaussian_laplace.txt'
average_performances, all_performances = sparsity_experiment(
    n_repeats=n_repeats, fractions_unknown=fractions_unknown, stratify_rows=stratify_rows,
    model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Sparsity performances")
plt.plot(fractions_unknown, average_performances['MSE'])
plt.ylim(0,4)