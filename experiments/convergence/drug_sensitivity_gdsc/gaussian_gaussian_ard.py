'''
Measure convergence on the GDSC drug sensitivity dataset, with the All Gaussian
model (multivariate posterior).
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian_ard import BMF_Gaussian_Gaussian_ARD
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50
from BMF_Priors.experiments.convergence.convergence_experiment import measure_convergence_time

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_gdsc_ic50()
model_class = BMF_Gaussian_Gaussian_ARD
settings = {
    'R': R, 
    'M': M, 
    'K': 20, 
    'hyperparameters': { 'alpha':1., 'beta':1., 'alpha0':1., 'beta0':1. }, 
    'init': 'random', 
    'iterations': 200,
}

times, performances = measure_convergence_time(model_class, settings)


''' Plot the times, and performance vs iterations. '''
plt.figure()
plt.title("Performance against average time")
plt.plot(times, performances['MSE'])
plt.ylim(0,2000)

plt.figure()
plt.title("Performance against iteration")
plt.plot(performances['MSE'])
plt.ylim(0,2000)