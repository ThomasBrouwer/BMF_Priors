'''
Measure sparsity experiment on the MovieLens 100K dataset, with 
the All Gaussian model (multivariate posterior) and Wishart prior.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian_wishart import BMF_Gaussian_Gaussian_Wishart
from BMF_Priors.data.movielens.load_data import load_movielens_100K
from BMF_Priors.experiments.sparsity.sparsity_experiment import sparsity_experiment

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_movielens_100K()
model_class = BMF_Gaussian_Gaussian_Wishart
n_repeats = 10
stratify_rows = False
fractions_known = [0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
fractions_unknown = [1. - v for v in fractions_known]
settings = {
    'R': R, 
    'M': M, 
    'K': 5,
    'hyperparameters': { 'alpha':1., 'beta':1., 'mu0':0., 'beta0':1., 'W0':1. }, 
    'init': 'random', 
    'iterations': 250,
    'burn_in': 200,
    'thinning': 1,
}
fout = './results/performances_gaussian_gaussian_wishart.txt'
average_performances, all_performances = sparsity_experiment(
    n_repeats=n_repeats, fractions_unknown=fractions_unknown, stratify_rows=stratify_rows,
    model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Sparsity performances")
plt.plot(fractions_unknown, average_performances['MSE'])
plt.ylim(0,4)