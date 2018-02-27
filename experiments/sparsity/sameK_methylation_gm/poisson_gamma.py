'''
Measure sparsity experiment on the methylation GM dataset, with 
the Poisson + Gamma model.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_poisson_gamma import BMF_Poisson_Gamma
from BMF_Priors.data.methylation.load_data import load_gene_body_methylation_integer
from BMF_Priors.experiments.sparsity.sparsity_experiment import sparsity_experiment

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_gene_body_methylation_integer()
model_class = BMF_Poisson_Gamma
n_repeats = 10
stratify_rows = False
fractions_unknown = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
settings = {
    'R': R, 
    'M': M, 
    'K': 5,
    'hyperparameters': { 'a':1., 'b':1. }, 
    'init': 'random', 
    'iterations': 500,
    'burn_in': 450,
    'thinning': 1,
}
fout = './results/performances_poisson_gamma.txt'
average_performances, all_performances = sparsity_experiment(
    n_repeats=n_repeats, fractions_unknown=fractions_unknown, stratify_rows=stratify_rows,
    model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Sparsity performances")
plt.plot(fractions_unknown, average_performances['MSE'])
plt.ylim(0,10)