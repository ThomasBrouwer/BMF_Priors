'''
Measure convergence on the GDSC drug sensitivity dataset, with the Poisson +
Dirichlet + Gamma model.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_poisson_gamma_dirichlet import BMF_Poisson_Gamma_Dirichlet
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.convergence.convergence_experiment import measure_convergence_time

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = BMF_Poisson_Gamma_Dirichlet
settings = {
    'R': R, 
    'M': M, 
    'K': 20, 
    'hyperparameters': { 'alpha':1., 'a':1., 'b':1. }, 
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
