'''
Measure runtime on the GDSC drug sensitivity dataset, with the All Gaussian
model (univariate posterior).
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_gaussian_gaussian_univariate import BMF_Gaussian_Gaussian_univariate
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.runtime.runtime_experiment import measure_runtime


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = BMF_Gaussian_Gaussian_univariate
values_K = [5, 10, 20, 50]
settings = {
    'R': R, 
    'M': M, 
    'hyperparameters': { 'alpha':1., 'beta':1., 'lamb':0.1 }, 
    'init': 'random', 
    'iterations': 100,
}
fout = './results/times_gaussian_gaussian_univariate.txt'

times_per_iteration = measure_runtime(values_K, model_class, settings, fout)
print zip(values_K, times_per_iteration)