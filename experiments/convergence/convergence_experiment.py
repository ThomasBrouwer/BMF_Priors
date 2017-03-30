'''
General methods for running the convergence experiments.
'''

import numpy

def measure_convergence_time(repeats, model_class, settings, fout_performances=None, fout_times=None):
    ''' Run the convergence experiment :repeats times. 
        Return (performances, times), giving average performances (MSE) and 
        timestamps across the 10 runs. Also store them in :fout_performances 
        and :fout_times if they are not None.
        
        Arguments: 
        - repeats -- number of times to repeat experiment and average across.
        - model_class -- the BMF class we should use.
        - settings -- dictionary {'R', 'M', 'K', 'hyperparameters', 'init', 'iterations'}.
        - fout_performances, fout_times -- strings giving location of output files.
    '''
    # Run the method :repeats times
    times_repeats = []
    performances_repeats = []
    for i in range(repeats):
        print "Convergence experiment. Repeat %s." % (i+1)
        R, M, K, hyperparameters = settings['R'], settings['M'], settings['K'], settings['hyperparameters']
        init, iterations = settings['init'], settings['iterations']
        BMF = model_class(R,M,K,hyperparameters) 
        BMF.initialise(init)
        BMF.run(iterations)
        times, performances = BMF.all_times, BMF.all_performances
        times_repeats.append(times), performances_repeats.append(performances['MSE'])
        
    # Compute averages and store in files
    times_average = list(numpy.average(times_repeats, axis=0))
    performances_average = list(numpy.average(performances_repeats, axis=0))
    if fout_performances and fout_times:
        open(fout_performances,'w').write("%s" % performances_average)
        open(fout_times,'w').write("%s" % times_average)
    return (performances_average, times_average)