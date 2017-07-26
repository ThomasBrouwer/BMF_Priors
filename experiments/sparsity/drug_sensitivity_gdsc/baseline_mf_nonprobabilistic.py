'''
Measure sparsity experiment on the GDSC drug sensitivity dataset, with the
non-probabilistic MF baseline.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.baseline_mf_nonprobabilistic import MF_Nonprobabilistic
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.sparsity.sparsity_experiment import sparsity_experiment

import matplotlib.pyplot as plt


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = MF_Nonprobabilistic
n_repeats = 10
stratify_rows = False
fractions_unknown = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
settings = {
    'R': R, 
    'M': M, 
    'K': 4,
    'hyperparameters': { 'exponential_prior': 0.1 }, 
    'init': 'random', 
    'iterations': 1000,
    'burn_in': 0,
    'thinning': 0,
}
fout = './results/performances_baseline_mf_nonprobabilistic.txt'
average_performances, all_performances = sparsity_experiment(
    n_repeats=n_repeats, fractions_unknown=fractions_unknown, stratify_rows=stratify_rows,
    model_class=model_class, settings=settings, fout=fout)


''' Plot the performance. '''
plt.figure()
plt.title("Sparsity performances")
plt.plot(fractions_unknown, average_performances['MSE'])
plt.ylim(0,1000)