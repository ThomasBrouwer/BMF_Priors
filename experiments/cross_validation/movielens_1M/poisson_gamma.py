'''
Run nested cross-validation experiment on the MovieLens 1M dataset, with 
the Poisson + Gamma model.
'''

project_location = "/home/tab43/Documents/Projects/libraries/" # "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.bmf_poisson_gamma import BMF_Poisson_Gamma
from BMF_Priors.code.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from BMF_Priors.data.movielens.load_data import load_processed_movielens_1M


''' Settings BMF model. '''
method = BMF_Poisson_Gamma
R, M = load_processed_movielens_1M()
hyperparameters = { 'a':1., 'b':1. }
train_config = {
    'iterations' : 120,
    'init' : 'random',
}
predict_config = {
    'burn_in' : 100,
    'thinning' : 1,
}


''' Settings nested cross-validation. '''
K_range = [1,2,3]
no_folds = 5
no_threads = 5
parallel = False
folder_results = './results/poisson_gamma/'
output_file = folder_results+'results.txt'
files_nested_performances = [folder_results+'fold_%s.txt'%(fold+1) for fold in range(no_folds)]


''' Construct the parameter search. '''
parameter_search = [{'K':K, 'hyperparameters':hyperparameters} for K in K_range]


''' Run the cross-validation framework. '''
nested_crossval = MatrixNestedCrossValidation(
    method=method,
    R=R,
    M=M,
    K=no_folds,
    P=no_threads,
    parameter_search=parameter_search,
    train_config=train_config,
    predict_config=predict_config,
    file_performance=output_file,
    files_nested_performances=files_nested_performances,
)
nested_crossval.run(parallel=parallel)