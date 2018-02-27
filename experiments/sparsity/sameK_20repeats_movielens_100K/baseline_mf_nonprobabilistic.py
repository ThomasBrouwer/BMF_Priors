'''
Measure sparsity experiment on the MovieLens 100K dataset, with the
non-probabilistic MF baseline.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BMF_Priors.code.models.baseline_mf_nonprobabilistic import MF_Nonprobabilistic
from BMF_Priors.data.movielens.load_data import load_processed_movielens_100K
from BMF_Priors.experiments.sparsity.sparsity_experiment import sparsity_experiment

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_processed_movielens_100K()
model_class = MF_Nonprobabilistic
n_repeats = 20
stratify_rows = False
fractions_known = [0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
fractions_unknown = [1. - v for v in fractions_known]
settings = {
    'R': R, 
    'M': M, 
    'K': 5,
    'hyperparameters': { 'exponential_prior': 0.1 }, 
    'init': 'random', 
    'iterations': 500,
    'burn_in': 0,
    'thinning': 0,
}
fout = './results/performances_baseline_mf_nonprobabilistic.txt'
average_performances, all_performances = sparsity_experiment(
    n_repeats=n_repeats, fractions_unknown=fractions_unknown, stratify_rows=stratify_rows,
    model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Sparsity performances")
plt.plot(fractions_unknown, average_performances['MSE'])
plt.ylim(0,4)