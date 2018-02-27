'''
Measure sparsity experiment on the GDSC drug sensitivity dataset, with 
the Poisson + Gamma model.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_poisson_gamma import BMF_Poisson_Gamma
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.sparsity.sparsity_experiment import sparsity_experiment

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = BMF_Poisson_Gamma
n_repeats = 10
stratify_rows = False
fractions_unknown = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
settings = {
    'R': R, 
    'M': M, 
    'K': 4,
    'hyperparameters': { 'a':1., 'b':1. }, 
    'init': 'random', 
    'iterations': 250,
    'burn_in': 200,
    'thinning': 2,
}
fout = './results/performances_poisson_gamma.txt'
average_performances, all_performances = sparsity_experiment(
    n_repeats=n_repeats, fractions_unknown=fractions_unknown, stratify_rows=stratify_rows,
    model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Sparsity performances")
plt.plot(fractions_unknown, average_performances['MSE'])
plt.ylim(0,1000)