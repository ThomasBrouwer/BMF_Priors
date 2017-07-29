'''
Measure runtime on the GDSC drug sensitivity dataset, with the non-probabilistic MF baseline.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.baseline_mf_nonprobabilistic import MF_Nonprobabilistic
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.runtime.runtime_experiment import measure_runtime


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = MF_Nonprobabilistic
values_K = [5, 10, 20, 50]
settings = {
    'R': R, 
    'M': M, 
    'hyperparameters': { 'exponential_prior': 0.1 }, 
    'init': 'exponential', 
    'iterations': 100,
}
fout = './results/times_baseline_mf_nonprobabilistic.txt'

times_per_iteration = measure_runtime(values_K, model_class, settings, fout)
print zip(values_K, times_per_iteration)