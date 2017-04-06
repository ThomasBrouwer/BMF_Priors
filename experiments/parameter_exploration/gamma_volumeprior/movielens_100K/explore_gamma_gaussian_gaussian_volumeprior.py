'''
Run cross-validation for the Gaussian+Gaussian+VP model with different gamma 
values, on the MovieLens data.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian_volumeprior import BMF_Gaussian_Gaussian_VolumePrior
from BMF_Priors.data.movielens.load_data import load_movielens_100K
from BMF_Priors.experiments.parameter_exploration.gamma_volumeprior.drug_sensitivity_gdsc.explore_gamma_gaussian_gaussian_volumeprior import explore_gamma

import itertools
import matplotlib.pyplot as plt


''' Run the experiment for the Gaussian + Gaussian + VP model. '''
R, M = load_movielens_100K()
model_class = BMF_Gaussian_Gaussian_VolumePrior
n_folds = 5
values_gamma = [10**-50, 10**-40, 10**-30, 10**-20, 10**-10, 10**0] 
values_K = [2, 5]
values_gamma_K = list(itertools.product(values_gamma, values_K))
settings = {
    'R': R, 
    'M': M, 
    'hyperparameters': { 'alpha':1., 'beta':1., 'lamb':0.1 }, 
    'init': 'random', 
    'iterations': 100,
    'burn_in': 90,
    'thinning': 1,
}
fout = './performances_gaussian_gaussian_volumeprior.txt'
average_performances = explore_gamma(
    n_folds=n_folds, values_gamma_K=values_gamma_K, model_class=model_class, settings=settings, fout=fout)


''' Plot the performances. '''
plt.figure()
plt.ylim(0,20)
plt.title("gamma exploration performances")
for K in values_K:
    performances = [perf for perf,(gamma,Kp) in zip(average_performances['MSE'],values_gamma_K) if Kp == K]
    plt.semilogx(values_gamma, performances, label=K)
    plt.legend(loc=2)
    plt.savefig('gaussian_gaussian_volumeprior.png')