'''
General methods for running the sparsity experiments.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.cross_validation.mask import try_generate_M_rows
from BMF_Priors.code.cross_validation.mask import try_generate_M_columns

import numpy

ATTEMPTS_GENERATE_FOLDS = 100
METRICS = ['MSE', 'R^2', 'Rp']

def sparsity_experiment(n_repeats, fractions_unknown, stratify_rows, model_class, settings, fout=None):
    ''' Run the sparsity experiment.
        For each fraction in :fractions_unknown, run the sparsity test :n_repeats
        times. We split the data randomly into :fractions_unknown missing values
        and 1-:fractions_unknown observed, and predict the missing ones.
        If stratify_rows = True, make sure each row (column if False) has at 
        least one entry.
        
        Return (average_performances, all_performances), both a dictionary 
        ('MSE', 'R^2, 'Rp'). The former gives the average performances for each
        value of K, and the latter gives all performances for each fold (list 
        of lists).
        Also store all_performances if :fout is not None.
        
        Arguments: 
        - n_repeats -- number of times to run sparsity experiment for each fraction.
        - fractions_unknown -- list of fractions to try.
        - stratify_rows -- whether to ensure one entry per row or column.
        - model_class -- the BMF class we should use.
        - settings -- dictionary {'R', 'M', 'K', 'hyperparameters', 'init', 'iterations', 'burn_in', 'thinning'}.
        - fout -- string giving location of output file.
    '''
    # Extract the settings
    R, M, K, hyperparameters = settings['R'], settings['M'], settings['K'], settings['hyperparameters']
    init, iterations = settings['init'], settings['iterations']
    burn_in, thinning = settings['burn_in'], settings['thinning']
        
    # Generate the folds
    generate_M = try_generate_M_rows if stratify_rows else try_generate_M_columns
    I, J = M.shape
    all_Ms_training_and_test = [
        [
            generate_M(I=I, J=J, fraction=fraction, attempts=ATTEMPTS_GENERATE_FOLDS, M=M)
            for r in range(n_repeats)
        ]
        for fraction in fractions_unknown
    ]
    all_performances = { metric:[] for metric in METRICS }
    for fraction, Ms_train_and_test in zip(fractions_unknown, all_Ms_training_and_test):
        # For each value of K, run the model on each fold and measure performances
        print "Sparsity experiment. Fraction=%s." % (fraction)
        performances = { metric:[] for metric in METRICS }
    
        for i, (M_train, M_test) in enumerate(Ms_train_and_test):
            print "Repeat %s for fraction=%s." % (i+1, fraction)
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