"""
Bayesian Matrix Factorisation with Poisson likelihood and Gamma priors.

Rij ~ N(Ui*Vj,tau^-1), tau ~ Gamma(alpha,beta), Uik ~ Gamma(a,b), Vjk ~ Gamma(a,b)

Random variables: U, V.
Hyperparameters: a, b.
"""

from bmf import BMF
from Gibbs.updates import update_Z_poisson
from Gibbs.updates import update_U_poisson_gamma
from Gibbs.updates import update_V_poisson_gamma
from Gibbs.initialise import initialise_Z_multinomial
from Gibbs.initialise import initialise_U_gamma

import numpy
import time

METRICS = ['MSE', 'R^2', 'Rp']
OPTIONS_INIT = ['random', 'exp']
DEFAULT_HYPERPARAMETERS = {
    'a': 1.,
    'b': 1.,
}

class BMF_Poisson_Gamma(BMF):
    def __init__(self,R,M,K,hyperparameters={}):
        """ Set up the class. """
        super(BMF_Poisson_Gamma, self).__init__(R, M, K)
        self.a = hyperparameters.get('a', DEFAULT_HYPERPARAMETERS['a'])
        self.b = hyperparameters.get('b', DEFAULT_HYPERPARAMETERS['b'])     
        
        
    def initialise(self,init):
        """ Initialise the values of the random variables in this model. """
        assert init in OPTIONS_INIT, \
            "Unknown initialisation option: %s. Should be one of %s." % (init, OPTIONS_INIT)
        self.U = initialise_U_gamma(init=init, I=self.I, K=self.K, a=self.a, b=self.b)
        self.V = initialise_U_gamma(init=init, I=self.J, K=self.K, a=self.a, b=self.b)
        self.Z = initialise_Z_multinomial(init=init, R=self.R, U=self.U, V=self.V)
        
        
    def run(self,iterations):
        """ Run the Gibbs sampler for the specified number of iterations. """
        self.all_U = numpy.zeros((iterations,self.I,self.K))  
        self.all_V = numpy.zeros((iterations,self.J,self.K))
        #self.all_Z = numpy.zeros((iterations,self.I,self.J,self.K))
        self.all_times = []
        self.all_performances = { metric: [] for metric in METRICS } 
        
        time_start = time.time()
        for it in range(iterations):
            # Update the random variables
            self.Z = update_Z_poisson(
                R=self.R, M=self.M, U=self.U, V=self.V)
            self.U = update_U_poisson_gamma(
                a=self.a, b=self.b, M=self.M, V=self.V, Z=self.Z)
            self.V = update_V_poisson_gamma(
                a=self.a, b=self.b, M=self.M, U=self.U, Z=self.Z)
            
            # Store the draws
            self.all_U[it], self.all_V[it] = numpy.copy(self.U), numpy.copy(self.V)
            #self.all_Z[it] = numpy.copy(self.Z)
            
            # Print the performance, store performance and time
            perf = self.predict_while_running()
            for metric in METRICS:
                self.all_performances[metric].append(perf[metric])
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)   
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
