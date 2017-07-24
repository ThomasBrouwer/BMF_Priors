'''
Measure noise experiment on the GDSC drug sensitivity dataset, with 
the Gaussian + Truncated Normal + hierarchical model.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_truncatednormal_hierarchical import BMF_Gaussian_TruncatedNormal_Hierarchical
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.noise.noise_experiment import noise_experiment

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = BMF_Gaussian_TruncatedNormal_Hierarchical
n_repeats = 10
stratify_rows = False
noise_to_signal_ratios = [1., 1.1, 1.2, 1.5, 2., 3., 6., 11.]
settings = {
    'R': R, 
    'M': M, 
    'K': 8,
    'hyperparameters': { 'alpha':1., 'beta':1., 'mu_mu':0., 'tau_mu':0.1, 'a':1., 'b':1. }, 
    'init': 'random', 
    'iterations': 250,
    'burn_in': 200,
    'thinning': 2,
}
fout = './results/performances_gaussian_truncatednormal_hierarchical.txt'
average_performances, all_performances = noise_experiment(
    n_repeats=n_repeats, noise_to_signal_ratios=noise_to_signal_ratios, 
    stratify_rows=stratify_rows, model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
variance = R.var()
average_performances_variance_ratio = [
     (NSR * variance) / avr_perf for (avr_perf, NSR) in zip(average_performances['MSE'], noise_to_signal_ratios)]
plt.figure()
plt.title("Noise performances")
plt.bar(range(len(noise_to_signal_ratios)), average_performances_variance_ratio)
plt.ylim(0,3)
plt.ylabel("Ratio of variance to predictive error")
plt.xticks(range(len(noise_to_signal_ratios)), [v-1 for v in noise_to_signal_ratios])
plt.xlabel("Noise added (noise to signal ratio)")