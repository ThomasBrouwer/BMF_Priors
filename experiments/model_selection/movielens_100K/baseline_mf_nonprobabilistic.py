'''
Measure model selection experiment on the GDSC drug sensitivity dataset, with 
the non-probabilistic MF baseline.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.baseline_mf_nonprobabilistic import MF_Nonprobabilistic
from BMF_Priors.data.movielens.load_data import load_processed_movielens_100K
from BMF_Priors.experiments.model_selection.model_selection_experiment import measure_model_selection

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_processed_movielens_100K()
model_class = MF_Nonprobabilistic
n_folds = 10
values_K = [1,2,3,4,6,8,10,15]
settings = {
    'R': R, 
    'M': M, 
    'hyperparameters': { 'exponential_prior': 0.1 }, 
    'init': 'exponential', 
    'iterations': 300,
    'burn_in': 0,
    'thinning': 0,
}
fout = './results/performances_baseline_mf_nonprobabilistic.txt'
average_performances, all_performances = measure_model_selection(
    n_folds=n_folds, values_K=values_K, model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Model selection performances")
plt.plot(values_K, average_performances['MSE'])
plt.ylim(0,20)