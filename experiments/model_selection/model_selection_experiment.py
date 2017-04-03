'''
General methods for running the model selection experiments.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.cross_validation.mask import compute_folds_attempts
from BMF_Priors.code.models.bmf_gaussian_gaussian_wishart import BMF_Gaussian_Gaussian_Wishart

import numpy

ATTEMPTS_GENERATE_FOLDS = 1000
METRICS = ['MSE', 'R^2', 'Rp']

def measure_model_selection(n_folds, values_K, model_class, settings, fout=None):
    ''' Run the model selection experiment.
        For each K in :values_K, run :n_folds cross-validation and measure the
        performances.
        Return (average_performances, all_performances), both a dictionary 
        ('MSE', 'R^2, 'Rp'). The former gives the average performances for each
        value of K, and the latter gives all performances for each fold (list 
        of lists).
        Also store all_performances if :fout is not None.
        
        Arguments: 
        - n_folds -- number of folds for cross-validation.
        - values_K -- range of values for K we should try
        - model_class -- the BMF class we should use.
        - settings -- dictionary {'R', 'M', 'hyperparameters', 'init', 'iterations', 'burn_in', 'thinning'}.
        - fout_performances, fout_times -- strings giving location of output files.
    '''
    # Extract the settings
    R, M, hyperparameters = settings['R'], settings['M'], settings['hyperparameters']
    init, iterations = settings['init'], settings['iterations']
    burn_in, thinning = settings['burn_in'], settings['thinning']
        
    # Generate the folds
    I, J = M.shape
    all_Ms_training_and_test = [
        compute_folds_attempts(I=I,J=J,no_folds=n_folds,attempts=ATTEMPTS_GENERATE_FOLDS,M=M)
        for K in values_K
    ]
    all_performances = { metric:[] for metric in METRICS }
    for K, (Ms_train, Ms_test) in zip(values_K, all_Ms_training_and_test):
        # For each value of K, run the model on each fold and measure performances
        print "Model selection experiment. K=%s." % (K)
        performances = { metric:[] for metric in METRICS }
    
        # If GGW model, override v0 value to K
        if model_class == BMF_Gaussian_Gaussian_Wishart:
            hyperparameters['v0'] = K    
    
        for i, (M_train, M_test) in enumerate(zip(Ms_train, Ms_test)):
            print "Fold %s for K=%s." % (i+1, K)
            BMF = model_class(R, M_train, K, hyperparameters) 
            BMF.initialise(init)
            BMF.run(iterations)
            performance = BMF.predict(M_pred=M_test, burn_in=burn_in, thinning=thinning)
            for metric in METRICS:
                performances[metric].append(performance[metric])
        for metric in METRICS:
            all_performances[metric].append(performances[metric])
    average_performances = { 
        metric : [numpy.mean(performances) for performances in all_performances[metric] ] 
        for metric in METRICS }
    if fout:
        open(fout,'w').write("%s" % all_performances)
    return (average_performances, all_performances)