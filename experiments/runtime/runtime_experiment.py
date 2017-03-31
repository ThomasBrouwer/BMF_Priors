'''
General methods for running the runtime speed experiments.
'''

def measure_runtime(values_K, model_class, settings, fout=None):
    ''' Run the runtime experiment, measuring the average time per iteration
        for different values of K. 
        Return (times), giving average runtime per iteration for each K in 
        :values_K. Also store them in :fout_performances and :fout_times if 
        they are not None.
        
        Arguments: 
        - values_K -- list of values for K we should try.
        - model_class -- the BMF class we should use.
        - settings -- dictionary {'R', 'M', hyperparameters', 'init', 'iterations'}.
        - fout -- strings giving location of output files.
    '''
    times_per_iteration = []
    for K in values_K:
        print "Runtime experiment. Trying K=%s." % (K)
        R, M, hyperparameters = settings['R'], settings['M'], settings['hyperparameters']
        init, iterations = settings['init'], settings['iterations']
        BMF = model_class(R,M,K,hyperparameters) 
        BMF.initialise(init)
        BMF.run(iterations)
        total_time = BMF.all_times[-1]
        average_time = total_time / float(iterations)
        times_per_iteration.append(average_time)
    if fout:
        open(fout,'w').write("%s" % times_per_iteration)
    return (times_per_iteration)