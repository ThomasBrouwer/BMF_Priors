'''
Measure convergence on the GDSC drug sensitivity dataset, with the Gaussian + 
Gaussian + Volume Prior (nonnegative) model.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian_volumeprior_nonnegative import BMF_Gaussian_Gaussian_VolumePrior_nonnegative
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.convergence.convergence_experiment import measure_convergence_time

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = BMF_Gaussian_Gaussian_VolumePrior_nonnegative
settings = {
    'R': R, 
    'M': M, 
    'K': 20, 
    'hyperparameters': { 'alpha':1., 'beta':1., 'lamb':0.1, 'gamma':10**-50 }, 
    'init': 'random', 
    'iterations': 200,
}
fout_performances = './results/performances_gaussian_gaussian_volumeprior_nonnegative.txt'
fout_times = './results/times_gaussian_gaussian_volumeprior_nonnegative.txt'
repeats = 10
performances, times = measure_convergence_time(
    repeats, model_class, settings, fout_performances, fout_times)


''' Plot the times, and performance vs iterations. '''
plt.figure()
plt.title("Performance against average time")
plt.plot(times, performances)
plt.ylim(0,2000)

plt.figure()
plt.title("Performance against iteration")
plt.plot(performances)
plt.ylim(0,2000)
