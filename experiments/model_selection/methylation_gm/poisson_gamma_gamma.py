'''
Measure model selection experiment on the methylation GM dataset, with 
Poisson likelihood, Gamma priors, and Gamma hierarchical priors.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_poisson_gamma_gamma import BMF_Poisson_Gamma_Gamma
from BMF_Priors.data.methylation.load_data import load_gene_body_methylation_integer
from BMF_Priors.experiments.model_selection.model_selection_experiment import measure_model_selection

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_gene_body_methylation_integer()
model_class = BMF_Poisson_Gamma_Gamma
n_folds = 10
values_K = [1,2,3,4,6,8,10,15,20,30]
settings = {
    'R': R, 
    'M': M, 
    'hyperparameters': { 'a':1., 'ap':1., 'bp':1. }, 
    'init': 'random', 
    'iterations': 500,
    'burn_in': 450,
    'thinning': 1,
}
fout = './results/performances_poisson_gamma_gamma.txt'
average_performances, all_performances = measure_model_selection(
    n_folds=n_folds, values_K=values_K, model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Model selection performances")
plt.plot(values_K, average_performances['MSE'])
plt.ylim(0,5)