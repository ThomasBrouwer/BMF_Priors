'''
Measure model selection experiment on the MovieLens 100K dataset, with 
the Gaussian + Laplace model.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_laplace import BMF_Gaussian_Laplace
from BMF_Priors.data.movielens.load_data import load_processed_movielens_100K
from BMF_Priors.experiments.model_selection.model_selection_experiment import measure_model_selection

import matplotlib.pyplot as plt
import math


''' Run the experiment. '''
R, M = load_processed_movielens_100K()
model_class = BMF_Gaussian_Laplace
n_folds = 10
values_K = [1,2,3,4,6,8,10,15]
settings = {
    'R': R, 
    'M': M, 
    'hyperparameters': { 'alpha':1., 'beta':1., 'eta':math.sqrt(10.) }, 
    'init': 'random', 
    'iterations': 200,
    'burn_in': 180,
    'thinning': 1,
}
fout = './results/performances_gaussian_laplace.txt'
average_performances, all_performances = measure_model_selection(
    n_folds=n_folds, values_K=values_K, model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Model selection performances")
plt.plot(values_K, average_performances['MSE'])
plt.ylim(0,2)
