'''
General methods for running the factor analysis experiments.
'''

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import numpy

def run_model_store_matrices(n_repeats, model_class, settings, fout_U=None, fout_V=None):
    ''' Run the factor analysis experiment.
        Run the model, :n_repeats times, and return a list of the expU and expV
        matrices: (all_expU, all_expV).
        Also store them if :fout_U, :fout_V is not None.
        
        Arguments: 
        - n_repeats -- number of times to run model and number of U, V matrices to return.
        - model_class -- the BMF class we should use.
        - settings -- dictionary {'R', 'M', 'K', 'hyperparameters', 'init', 'iterations', 'burn_in', 'thinning'}.
        - fout_U, fout_V -- strings giving locations of output files.
    '''
    # Extract the settings
    R, M, K, hyperparameters = settings['R'], settings['M'], settings['K'], settings['hyperparameters']
    init, iterations = settings['init'], settings['iterations']
    burn_in, thinning = settings['burn_in'], settings['thinning']
    I, J = R.shape[0], R.shape[1]
    
    # For each repeat, approximate U,V and store them
    all_expU, all_expV = numpy.zeros((n_repeats,I,K)), numpy.zeros((n_repeats,J,K))
    for r in range(n_repeats):
        print "Repeat %s." % (r+1)
        BMF = model_class(R, M, K, hyperparameters) 
        BMF.initialise(init)
        BMF.run(iterations)
        expU, expV = BMF.approx_expectation_UV(burn_in=burn_in, thinning=thinning)
        all_expU[r,:,:], all_expV[r,:,:] = expU, expV
        
    if fout_U and fout_V:
        open(fout_U,'w').write("%s" % all_expU.tolist())
        open(fout_V,'w').write("%s" % all_expV.tolist())
    return (all_expU, all_expV)