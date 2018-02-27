'''
Measure noise experiment on the GDSC drug sensitivity dataset, with 
the Gaussian + Truncated Normal + hierarchical model.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_truncatednormal_hierarchical import BMF_Gaussian_TruncatedNormal_Hierarchical
from BMF_Priors.experiments.noise.movielens_100K.data.create_load_noise_datasets import load_noise_datasets
from BMF_Priors.experiments.noise.noise_experiment import noise_experiment

import matplotlib.pyplot as plt


''' Run the experiment. '''
noise_to_signal_ratios, Rs_noise, M = load_noise_datasets()
model_class = BMF_Gaussian_TruncatedNormal_Hierarchical
n_repeats = 10
stratify_rows = False
settings = {
    'M': M, 
    'K': 2,
    'hyperparameters': { 'alpha':1., 'beta':1., 'mu_mu':0., 'tau_mu':0.1, 'a':1., 'b':1. }, 
    'init': 'random', 
    'iterations': 100,
    'burn_in': 80,
    'thinning': 1,
}
fout = './results/performances_gaussian_truncatednormal_hierarchical.txt'
average_performances, all_performances = noise_experiment(
    n_repeats=n_repeats, noise_to_signal_ratios=noise_to_signal_ratios, Rs_noise=Rs_noise,
    stratify_rows=stratify_rows, model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
average_performances_variance_ratio = [
     R_noise.var() / avr_perf 
     for (avr_perf, R_noise, NSR) in zip(average_performances['MSE'], Rs_noise, noise_to_signal_ratios)
]
plt.figure()
plt.title("Noise performances")
plt.bar(range(len(noise_to_signal_ratios)), average_performances_variance_ratio)
plt.ylim(0,3)
plt.ylabel("Ratio of data variance to predictive error")
plt.xticks(range(len(noise_to_signal_ratios)), noise_to_signal_ratios)
plt.xlabel("Noise added (noise to signal ratio)")