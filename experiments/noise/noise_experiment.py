'''
General methods for running the noise experiments.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.cross_validation.mask import try_generate_M_rows
from BMF_Priors.code.cross_validation.mask import try_generate_M_columns
from BMF_Priors.code.models.Gibbs.distributions.normal import normal_draw

import itertools
import numpy

ATTEMPTS_GENERATE_FOLDS = 100
METRICS = ['MSE', 'R^2', 'Rp']
FRACTION_TRAIN = 0.9

def noise_experiment(n_repeats, noise_to_signal_ratios, stratify_rows, model_class, settings, fout=None):
    ''' Run the noise experiment.
        For each noise-to-signal ratio in :noise_to_signal_ratios, run the noise
        test :n_repeats times. We add a level of Gaussian noise to the dataset
        with variance equal to :noise_to_signal_ratios times the variance of the
        data. We then split the data randomly into 90% train and 10% test, and 
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
        - noise_to_signal_ratios -- list of noise-to-signal ratios for noise levels.
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
    for NSR, Ms_train_and_test in zip(noise_to_signal_ratios, all_Ms_training_and_test):
        # For each noise-to-signal ratio, run the model on each fold and measure performances
        print "Noise experiment. Noise-to-signal ratio=%s." % (NSR)
        performances = { metric:[] for metric in METRICS }
    
        for i, (M_train, M_test) in enumerate(Ms_train_and_test):
            print "Repeat %s for NSR=%s." % (i+1, NSR)
            R_noise = add_noise(R=R, NSR=NSR)
            print "Variance of R with noise=%s." % (R_noise.var())
            
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


def add_noise(R, NSR):
    ''' Method for adding noise to a dataset. 
        Also cast values to integers, and make nonnegative values zero.
        If NSR is 5., add 4 times existing variance to end up with 5. '''
    variance_signal = R.var()
    tau = 1. / (variance_signal * (NSR-1))
    print "Noise: %s%%. Variance in dataset is %s. Adding noise with variance %s." % (100.*NSR,variance_signal,1./tau)
    
    if numpy.isinf(tau):
        return numpy.copy(R)
    (I,J) = R.shape
    R_noise = numpy.zeros((I,J))
    for i,j in itertools.product(range(I),range(J)):
        new_Rij = normal_draw(R[i,j],tau)
        R_noise[i,j] = max(0, int(new_Rij))
    return R_noise