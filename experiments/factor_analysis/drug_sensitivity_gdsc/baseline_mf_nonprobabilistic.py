'''
Run factor analysis experiment on the GDSC drug sensitivity dataset, with the
non-probabilistic MF baseline.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BMF_Priors.code.models.baseline_mf_nonprobabilistic import MF_Nonprobabilistic
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.factor_analysis.factor_analysis import run_model_store_matrices


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = MF_Nonprobabilistic
n_repeats = 10
settings = {
    'R': R, 
    'M': M, 
    'K': 10,
    'hyperparameters': { 'exponential_prior': 0.1 }, 
    'init': 'random', 
    'iterations': 500,
    'burn_in': 0,
    'thinning': 0,
}
fout_U = './results/baseline_mf_nonprobabilistic_U.txt'
fout_V = './results/baseline_mf_nonprobabilistic_V.txt'
all_expU, all_expV = run_model_store_matrices(
    n_repeats=n_repeats, model_class=model_class, settings=settings, fout_U=fout_U, fout_V=fout_V)
