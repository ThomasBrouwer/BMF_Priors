"""
NOTE: This model's posteriors for the Dirichlet part are not conjugate and hence do not work.

Bayesian Matrix Factorisation with Poisson likelihood, Dirichlet prior over U,
and Gamma prior over V.

Rij ~ N(Ui*Vj,tau^-1), tau ~ Gamma(alpha,beta), Ui ~ Dir(alpha), Vjk ~ Gamma(a,b)

Random variables: U, V.
Hyperparameters: alpha, a, b.

If alpha is given as an integer, make it a K-dimensional vector with that value.
"""

from bmf import BMF
from Gibbs.updates import update_Z_poisson
from Gibbs.updates import update_U_poisson_dirichlet
from Gibbs.updates import update_V_poisson_gamma
from Gibbs.initialise import initialise_Z_multinomial
from Gibbs.initialise import initialise_U_gamma
from Gibbs.initialise import initialise_U_dirichlet

import numpy
import time

METRICS = ['MSE', 'R^2', 'Rp']
OPTIONS_INIT = ['random', 'exp']
DEFAULT_HYPERPARAMETERS = {
    'alpha': 1.,
    'a': 1.,
    'b': 1.,
}

class BMF_Poisson_Gamma_Dirichlet(BMF):
    def __init__(self,R,M,K,hyperparameters={}):
        """ Set up the class. """
        super(BMF_Poisson_Gamma_Dirichlet, self).__init__(R, M, K)
        self.alpha = hyperparameters.get('alpha', DEFAULT_HYPERPARAMETERS['alpha'])
        self.a =     hyperparameters.get('a',     DEFAULT_HYPERPARAMETERS['a'])
        self.b =     hyperparameters.get('b',     DEFAULT_HYPERPARAMETERS['b'])     
        self.alpha = self.alpha if isinstance(self.alpha, numpy.ndarray)  \
                     else self.alpha * numpy.ones(K)
        assert self.alpha.shape == (K,), "alpha should be shape (%s,), not %s." % (
            self.K, self.alpha.shape)
        
        
    def initialise(self,init):
        """ Initialise the values of the random variables in this model. """
        assert init in OPTIONS_INIT, \
            "Unknown initialisation option: %s. Should be one of %s." % (init, OPTIONS_INIT)
        self.U = initialise_U_dirichlet(init=init, I=self.I, K=self.K, alpha=self.alpha)
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
            self.U = update_U_poisson_dirichlet(
                alpha=self.alpha, M=self.M, Z=self.Z)
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
