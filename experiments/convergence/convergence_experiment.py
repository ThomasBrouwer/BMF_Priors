'''
General methods for running the convergence experiments.
'''

def measure_convergence_time(model_class, settings):
    ''' Run the convergence experiment. Return (times, performances).
        model_class -- the BMF class we should use.
        settings -- dictionary {'R', 'M', 'K', 'hyperparameters', 'init', 'iterations'}
    '''
    R, M, K, hyperparameters = settings['R'], settings['M'], settings['K'], settings['hyperparameters']
    init, iterations = settings['init'], settings['iterations']
    BMF = model_class(R,M,K,hyperparameters) 
    BMF.initialise(init)
    BMF.run(iterations)
    times, performances = BMF.all_times, BMF.all_performances
    return times, performances
