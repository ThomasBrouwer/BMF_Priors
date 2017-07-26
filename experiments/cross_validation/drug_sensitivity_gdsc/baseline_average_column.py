'''
Run cross-validation experiment on the GDSC drug sensitivity dataset, with the 
column-average baseline.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.baseline_average_column import ColumnAverage
from BMF_Priors.code.cross_validation.matrix_single_cross_validation import MatrixSingleCrossValidation
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer


''' Settings BMF model. '''
method = ColumnAverage
R, M = load_gdsc_ic50_integer()
hyperparameters = {}
train_config = { 'iterations': 0, 'init': '' }
predict_config = { 'burn_in': 0, 'thinning': 0 }
parameters = {'K':0, 'hyperparameters':hyperparameters}


''' Settings nested cross-validation. '''
no_folds = 5
folder_results = './results/baseline_average_column/'
output_file = folder_results+'results.txt'


''' Run the cross-validation framework. '''
crossval = MatrixSingleCrossValidation(
    method=method,
    R=R,
    M=M,
    K=no_folds,
    parameters=parameters,
    train_config=train_config,
    predict_config=predict_config,
    file_performance=output_file,
)
crossval.run()