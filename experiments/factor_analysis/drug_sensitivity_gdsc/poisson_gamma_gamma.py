'''
Run factor analysis experiment on the GDSC drug sensitivity dataset, with 
Poisson likelihood, Gamma priors, and Gamma hierarchical priors.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_poisson_gamma_gamma import BMF_Poisson_Gamma_Gamma
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer
from BMF_Priors.experiments.factor_analysis.factor_analysis import run_model_store_matrices


''' Run the experiment. '''
R, M = load_gdsc_ic50_integer()
model_class = BMF_Poisson_Gamma_Gamma
n_repeats = 10
settings = {
    'R': R, 
    'M': M, 
    'K': 10,
    'hyperparameters': { 'a':1., 'ap':1., 'bp':1. }, 
    'init': 'random', 
    'iterations': 300,
    'burn_in': 200,
    'thinning': 2,
}
fout_U = './results/poisson_gamma_gamma_U.txt'
fout_V = './results/poisson_gamma_gamma_V.txt'
all_expU, all_expV = run_model_store_matrices(
    n_repeats=n_repeats, model_class=model_class, settings=settings, fout_U=fout_U, fout_V=fout_V)
