"""
Bayesian Matrix Factorisation with Gaussian likelihood and Truncated Normal priors.

Rij ~ N(Ui*Vj,tau^-1), tau ~ Gamma(alpha,beta), Uik ~ TN(muUV,tauUV), Vjk ~ TN(muUV,tauUV)

Random variables: U, V, tau.
Hyperparameters: alpha, beta, muUV, tauUV.
"""

from bmf import BMF
from Gibbs.updates import update_tau_gaussian
from Gibbs.updates import update_U_gaussian_truncatednormal
from Gibbs.updates import update_V_gaussian_truncatednormal
from Gibbs.initialise import initialise_tau_gamma
from Gibbs.initialise import initialise_U_truncatednormal

import numpy
import time

METRICS = ['MSE', 'R^2', 'Rp']
OPTIONS_INIT = ['random', 'exp']
DEFAULT_HYPERPARAMETERS = {
    'alpha': 1.,
    'beta': 1.,
    'muUV': 0.1,
    'tauUV': 0.1,
}

class BMF_Gaussian_TruncatedNormal(BMF):
    def __init__(self,R,M,K,hyperparameters={}):
        """ Set up the class. """
        super(BMF_Gaussian_TruncatedNormal, self).__init__(R, M, K)
        self.alpha = hyperparameters.get('alpha', DEFAULT_HYPERPARAMETERS['alpha'])
        self.beta =  hyperparameters.get('beta',  DEFAULT_HYPERPARAMETERS['beta'])   
        self.muUV =  hyperparameters.get('muUV',  DEFAULT_HYPERPARAMETERS['muUV']) 
        self.tauUV = hyperparameters.get('tauUV', DEFAULT_HYPERPARAMETERS['tauUV'])    
        
        
    def initialise(self,init):
        """ Initialise the values of the random variables in this model. """
        assert init in OPTIONS_INIT, \
            "Unknown initialisation option: %s. Should be one of %s." % (init, OPTIONS_INIT)
        self.U = initialise_U_truncatednormal(
            init=init, I=self.I, K=self.K, mu=self.muUV, tau=self.tauUV)
        self.V = initialise_U_truncatednormal(
            init=init, I=self.J, K=self.K, mu=self.muUV, tau=self.tauUV)
        self.tau = initialise_tau_gamma(
            alpha=self.alpha, beta=self.beta, R=self.R, M=self.M, U=self.U, V=self.V)
        
        
    def run(self,iterations):
        """ Run the Gibbs sampler for the specified number of iterations. """
        self.all_U = numpy.zeros((iterations,self.I,self.K))  
        self.all_V = numpy.zeros((iterations,self.J,self.K))   
        self.all_tau = numpy.zeros(iterations) 
        self.all_times = []
        self.all_performances = { metric: [] for metric in METRICS } 
        
        time_start = time.time()
        for it in range(iterations):
            # Update the random variables
            self.U = update_U_gaussian_truncatednormal(
                muU=self.muUV, tauU=self.tauUV, R=self.R, M=self.M, U=self.U, V=self.V, tau=self.tau) 
            self.V = update_V_gaussian_truncatednormal(
                muV=self.muUV, tauV=self.tauUV, R=self.R, M=self.M, U=self.U, V=self.V, tau=self.tau)
            self.tau = update_tau_gaussian(
                alpha=self.alpha, beta=self.beta, R=self.R, M=self.M, U=self.U, V=self.V)
            
            # Store the draws
            self.all_U[it], self.all_V[it] = numpy.copy(self.U), numpy.copy(self.V)
            self.all_tau[it] = self.tau
            
            # Print the performance, store performance and time
            perf = self.predict_while_running()
            for metric in METRICS:
                self.all_performances[metric].append(perf[metric])
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)   
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
