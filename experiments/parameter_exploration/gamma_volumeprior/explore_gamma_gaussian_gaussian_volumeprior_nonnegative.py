'''
Run cross-validation for the Gaussian+Gaussian+VP model with different gamma 
values, on the drug sensitivity data.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian_volumeprior_nonnegative import BMF_Gaussian_Gaussian_VolumePrior_nonnegative
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from explore_gamma_gaussian_gaussian_volumeprior import explore_gamma

import itertools
import matplotlib.pyplot as plt


''' Run the experiment for the Gaussian + Gaussian + VP (nonnegative) model. '''
R, M = load_gdsc_ic50_integer()
model_class = BMF_Gaussian_Gaussian_VolumePrior_nonnegative
n_folds = 5
values_gamma = [10**-40, 10**-35, 10**-30, 10**-25, 10**-20, 10**-15, 10**-10, 10**-5, 10**0, 10**5] 
values_K = [5, 10] #, 15]
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
fout = './performances_gaussian_gaussian_volumeprior_nonnegative.txt'
average_performances = explore_gamma(
    n_folds=n_folds, values_gamma_K=values_gamma_K, model_class=model_class, settings=settings, fout=fout)


''' Plot the performances. '''
plt.figure()
plt.ylim(600,1000)
plt.title("gamma exploration performances")
for K in values_K:
    performances = [perf for perf,(gamma,Kp) in zip(average_performances['MSE'],values_gamma_K) if Kp == K]
    plt.semilogx(values_gamma, performances, label=K)
    plt.legend(loc=2)
    plt.savefig('gaussian_gaussian_volumeprior_nonnegative.png')