'''
Measure runtime on the GDSC drug sensitivity dataset, with the Gaussian + 
Gaussian + Volume Prior (nonnegative) model.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian_volumeprior_nonnegative import BMF_Gaussian_Gaussian_VolumePrior_nonnegative
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.runtime.runtime_experiment import measure_runtime


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = BMF_Gaussian_Gaussian_VolumePrior_nonnegative
values_K = [5, 10, 20, 50]
settings = {
    'R': R, 
    'M': M, 
    'hyperparameters': { 'alpha':1., 'beta':1., 'lamb':0.1, 'gamma':10**-50 }, 
    'init': 'random', 
    'iterations': 100,
}
fout = './results/times_gaussian_gaussian_volumeprior_nonnegative.txt'

times_per_iteration = measure_runtime(values_K, model_class, settings, fout)
print zip(values_K, times_per_iteration)