'''
Run cross-validation for the Gaussian+Gaussian+VP model with different gamma 
values, on the drug sensitivity data.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.cross_validation.mask import compute_folds_attempts
from BMF_Priors.code.models.bmf_gaussian_gaussian_volumeprior import BMF_Gaussian_Gaussian_VolumePrior
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer

import numpy
import itertools
import matplotlib.pyplot as plt

ATTEMPTS_GENERATE_FOLDS = 1000
METRICS = ['MSE', 'R^2', 'Rp']

def explore_gamma(n_folds, values_gamma_K, model_class, settings, fout=None):
    ''' Try different values for gamma.
        Return (performances), giving average performances (MSE) for the gamma
        values in :n_folds cross-validation. Also store them if :fout is not None.
        
        Arguments: 
        - n_folds -- number of folds for cross-validation.
        - values_gamma_K -- values to try for gamma and K: list of tuples (gamma,K).
        - model_class -- the BMF class we should use.
        - settings -- dictionary {'R', 'M', 'K', 'hyperparameters', 'init', 'iterations', 'burn_in', 'thinning'}.
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
        for (gamma,K) in values_gamma_K
    ]
    all_performances = { metric:[] for metric in METRICS }

    # Run the cross-validations
    for (gamma, K), (Ms_train, Ms_test) in zip(values_gamma_K, all_Ms_training_and_test):
        # For each value of K, run the model on each fold and measure performances
        hyperparameters['gamma'] = gamma
        print "Parameter search experiment. gamma=%s, K=%s." % (gamma, K)
        performances = { metric:[] for metric in METRICS }
        for i, (M_train, M_test) in enumerate(zip(Ms_train, Ms_test)):
            print "Fold %s for gamma=%s, K=%s." % (i+1, gamma, K)
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
        open(fout,'w').write("%s" % average_performances)
    return (average_performances)
        

if __name__ == '__main__':
    ''' Run the experiment for the Gaussian + Gaussian + VP model. '''
    R, M = load_gdsc_ic50_integer()
    model_class = BMF_Gaussian_Gaussian_VolumePrior
    n_folds = 5
    values_gamma = [10**-40, 10**-35, 10**-30, 10**-25, 10**-20, 10**-15, 10**-10, 10**-5, 10**0, 10**5]
    values_K = [5, 10] #, 15]
    values_gamma_K = list(itertools.product(values_gamma, values_K))
    settings = {
        'R': R, 
        'M': M, 
        'hyperparameters': { 'alpha':1., 'beta':1., 'lamb':0.1 }, 
        'init': 'random', 
        'iterations': 100,
        'burn_in': 90,
        'thinning': 1,
    }
    fout = './performances_gaussian_gaussian_volumeprior.txt'
    average_performances = explore_gamma(
        n_folds=n_folds, values_gamma_K=values_gamma_K, model_class=model_class, settings=settings, fout=fout)
    
    
    ''' Plot the performances. '''
    plt.figure()
    plt.ylim(600,1000)
    plt.title("gamma exploration performances")
    for K in values_K:
        performances = [perf for perf,(gamma,Kp) in zip(average_performances['MSE'],values_gamma_K) if Kp == K]
        plt.semilogx(values_gamma, performances, label=K)
        plt.legend(loc=2)
        plt.savefig('gaussian_gaussian_volumeprior.png')