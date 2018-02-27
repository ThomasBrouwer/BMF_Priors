'''
General methods for running the noise experiments.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from BMF_Priors.code.cross_validation.mask import try_generate_M_rows
from BMF_Priors.code.cross_validation.mask import try_generate_M_columns

import numpy

ATTEMPTS_GENERATE_FOLDS = 100
METRICS = ['MSE', 'R^2', 'Rp']
FRACTION_TRAIN = 0.9

def noise_experiment(n_repeats, Rs_noise, noise_to_signal_ratios, stratify_rows, model_class, settings, fout=None):
    ''' Run the noise experiment.
        For each noise-to-signal ratio in :noise_to_signal_ratios, run the noise
        test :n_repeats times. We use the R in :Rs_noise, which has added Gaussian 
        noise. We then split the data randomly into 90% train and 10% test, and 
        predict the missing ones.
        If stratify_rows = True, make sure each row (column if False) has at 
        least one entry.
        
        Return (average_performances, all_performances), both a dictionary 
        ('MSE', 'R^2, 'Rp'). The former gives the average performances for each
        noise-to-signal ratio, and the latter gives all performances for each 
        fold (list of lists).
        Also store all_performances if :fout is not None.
        
        Arguments: 
        - n_repeats -- number of times to run sparsity experiment for each fraction.
        - Rs_noise -- list of the R datasets, with noise added.
        - noise_to_signal_ratios -- list of noise-to-signal ratios for noise levels.
        - stratify_rows -- whether to ensure one entry per row or column.
        - model_class -- the BMF class we should use.
        - settings -- dictionary {'M', 'K', 'hyperparameters', 'init', 'iterations', 'burn_in', 'thinning'}.
        - fout -- string giving location of output file.
    '''
    assert len(Rs_noise) == len(noise_to_signal_ratios), "Rs_noise should be of the same length as noise_to_signal_ratios!"
    
    # Extract the settings
    M, K, hyperparameters = settings['M'], settings['K'], settings['hyperparameters']
    init, iterations = settings['init'], settings['iterations']
    burn_in, thinning = settings['burn_in'], settings['thinning']
        
    # Generate the folds
    generate_M = try_generate_M_rows if stratify_rows else try_generate_M_columns
    I, J = M.shape
    fraction_observed = M.sum() / float(I*J)
    fraction_observed_after = fraction_observed * FRACTION_TRAIN
    fraction_missing_after = 1. - fraction_observed_after
    all_Ms_training_and_test = [
        [
            generate_M(I=I, J=J, fraction=fraction_missing_after, attempts=ATTEMPTS_GENERATE_FOLDS, M=M)
            for r in range(n_repeats)
        ]
        for NSR in noise_to_signal_ratios
    ]
        
    all_performances = { metric:[] for metric in METRICS }
    for R_noise, NSR, Ms_train_and_test in zip(Rs_noise, noise_to_signal_ratios, all_Ms_training_and_test):
        # For each noise-to-signal ratio, run the model on each fold and measure performances
        print "Noise experiment. Noise-to-signal ratio=%s. Variance R with noise=%s." % (NSR, R_noise.var())
        performances = { metric:[] for metric in METRICS }
    
        for i, (M_train, M_test) in enumerate(Ms_train_and_test):
            print "Repeat %s for NSR=%s." % (i+1, NSR)
            
            BMF = model_class(R_noise, M_train, K, hyperparameters) 
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