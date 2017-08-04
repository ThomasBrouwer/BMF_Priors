'''
Measure model selection experiment on the GDSC drug sensitivity dataset, with 
the Gaussian + Laplace + IG model.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_laplace_inversegaussian import BMF_Gaussian_Laplace_IG
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.model_selection.model_selection_experiment import measure_model_selection

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = BMF_Gaussian_Laplace_IG
n_folds = 10
values_K = [1,2,3,4,6,8,10,15,20,30]
settings = {
    'R': R, 
    'M': M, 
    'hyperparameters': { 'alpha':1., 'beta':1. }, 
    'init': 'random', 
    'iterations': 250,
    'burn_in': 200,
    'thinning': 1,
}
fout = './results/performances_gaussian_laplace_ig.txt'
average_performances, all_performances = measure_model_selection(
    n_folds=n_folds, values_K=values_K, model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Model selection performances")
plt.plot(values_K, average_performances['MSE'])
plt.ylim(0,1000)