'''
Measure sparsity experiment on the MovieLens 100K dataset, with the row-average 
baseline.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.baseline_average_row import RowAverage
from BMF_Priors.data.movielens.load_data import load_processed_movielens_100K
from BMF_Priors.experiments.sparsity.sparsity_experiment import sparsity_experiment

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_processed_movielens_100K()
model_class = RowAverage
n_repeats = 10
stratify_rows = False
fractions_known = [0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
fractions_unknown = [1. - v for v in fractions_known]
settings = {
    'R': R, 
    'M': M, 
    'K': 0,
    'hyperparameters': {}, 
    'init': 'random', 
    'iterations': 0,
    'burn_in': 0,
    'thinning': 0,
}
fout = './results/performances_baseline_average_row.txt'
average_performances, all_performances = sparsity_experiment(
    n_repeats=n_repeats, fractions_unknown=fractions_unknown, stratify_rows=stratify_rows,
    model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Sparsity performances")
plt.plot(fractions_unknown, average_performances['MSE'])
plt.ylim(0,4)