'''
Run cross-validation for the Gaussian+L21 model with different lambda values, 
on the drug sensitivity data.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from BMF_Priors.data.drug_sensitivity.load_data import load_ccle_ec50_integer
from BMF_Priors.experiments.parameter_exploration.lambda_l21.drug_sensitivity_gdsc.explore_lambda_gaussian_l21 import explore_lambda
from BMF_Priors.code.models.bmf_gaussian_l21 import BMF_Gaussian_L21

import itertools
import matplotlib.pyplot as plt


''' Run the experiment for the Gaussian + L21. '''
R, M = load_ccle_ec50_integer()
model_class = BMF_Gaussian_L21
n_folds = 5
values_lambda = [10**-25, 10**-20, 10**-15, 10**-10, 10**-5, 10**0, 10**5]
values_K = [2, 5]
values_lambda_K = list(itertools.product(values_lambda, values_K))
settings = {
    'R': R, 
    'M': M, 
    'hyperparameters': { 'alpha':1., 'beta':1., 'lamb':0.1 }, 
    'init': 'random', 
    'iterations': 250,
    'burn_in': 200,
    'thinning': 1,
}
fout = './performances_gaussian_gaussian_volumeprior.txt'
average_performances = explore_lambda(
    n_folds=n_folds, values_lambda_K=values_lambda_K, model_class=model_class, settings=settings, fout=fout)


''' Plot the performances. '''
plt.figure()
plt.ylim(0,20)
plt.title("lambda exploration performances")
for K in values_K:
    performances = [perf for perf,(lamb,Kp) in zip(average_performances['MSE'],values_lambda_K) if Kp == K]
    plt.semilogx(values_lambda, performances, label=K)
    plt.legend(loc=2)
    plt.savefig('gaussian_l21.png')