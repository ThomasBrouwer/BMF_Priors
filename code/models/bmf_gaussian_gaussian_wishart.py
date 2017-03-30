"""
Bayesian Matrix Factorisation with Gaussian likelihood, Gaussian priors, and 
Wishart prior.

Rij ~ N(Ui*Vj,tau^-1),                 tau ~ Gamma(alpha,beta), 
Ui ~ N(muU, sigmaU),                   Vj ~ N(muU, sigmaU),
muU, sigmaU ~ NIW(mu0, beta0, v0, W0), muV, sigmaV ~ NIW(mu0, beta0, v0, W0)

Random variables: U, V, muU, sigmaU, muV, sigmaV.
Hyperparameters: alpha, beta, mu0 (vector), beta0, v0, W0 (matrix).

If mu0 is given as an integer, make it a K-dimensional vector with that value.
if W0 is given as an integer, make it a K by K diagonal matrix with that value.
"""

from bmf import BMF
from Gibbs.updates import update_tau_gaussian
from Gibbs.updates import update_U_gaussian_gaussian_wishart
from Gibbs.updates import update_V_gaussian_gaussian_wishart
from Gibbs.updates import update_muU_sigmaU_gaussian_gaussian_wishart
from Gibbs.updates import update_muV_sigmaV_gaussian_gaussian_wishart
from Gibbs.initialise import initialise_tau_gamma
from Gibbs.initialise import initialise_U_gaussian_wishart
from Gibbs.initialise import initialise_muU_sigmaU_wishart

import numpy
import time

METRICS = ['MSE', 'R^2', 'Rp']
OPTIONS_INIT = ['random', 'exp']
DEFAULT_HYPERPARAMETERS = {
    'alpha': 1.,
    'beta': 1.,
    'mu0': 0.,
    'beta0': 1.,
    'v0': 1.,
    'W0': 1.,
}

class BMF_Gaussian_Gaussian_Wishart(BMF):
    def __init__(self,R,M,K,hyperparameters={}):
        """ Set up the class. """
        super(BMF_Gaussian_Gaussian_Wishart, self).__init__(R, M, K)
        self.alpha = hyperparameters.get('alpha', DEFAULT_HYPERPARAMETERS['alpha'])
        self.beta =  hyperparameters.get('beta',  DEFAULT_HYPERPARAMETERS['beta'])   
        self.mu0 =   hyperparameters.get('mu0',   DEFAULT_HYPERPARAMETERS['mu0'])
        self.beta0 = hyperparameters.get('beta0', DEFAULT_HYPERPARAMETERS['beta0'])
        self.v0 =    hyperparameters.get('v0',    DEFAULT_HYPERPARAMETERS['v0'])
        self.W0 =    hyperparameters.get('W0',    DEFAULT_HYPERPARAMETERS['W0'])
        self.mu0 = self.mu0 if isinstance(self.mu0, numpy.ndarray)  \
                   else self.mu0 * numpy.ones(K)
        self.W0 =  self.W0 if isinstance(self.W0, numpy.ndarray) \
                   else self.W0 * numpy.eye(K)
        assert self.mu0.shape == (K,), "mu0 should be shape (%s,), not %s." % (
            self.K, self.mu0.shape)
        assert self.W0.shape == (K,K), "W0 should be shape (%s,%s), not %s." % (
            self.K, self.K, self.W0.shape)
        assert self.v0 > self.K - 1, "v0 = %s should be greater than K - 1 = %s." % (self.v0, self.K-1)
        
        
    def initialise(self,init):
        """ Initialise the values of the random variables in this model. """
        assert init in OPTIONS_INIT, \
            "Unknown initialisation option: %s. Should be one of %s." % (init, OPTIONS_INIT)
        self.muU, self.sigmaU = initialise_muU_sigmaU_wishart(
            init=init, mu0=self.mu0, beta0=self.beta0, v0=self.v0, W0=self.W0)
        self.muV, self.sigmaV = initialise_muU_sigmaU_wishart(
            init=init, mu0=self.mu0, beta0=self.beta0, v0=self.v0, W0=self.W0)
        self.U = initialise_U_gaussian_wishart(init=init, I=self.I, K=self.K, muU=self.muU, sigmaU=self.sigmaU)
        self.V = initialise_U_gaussian_wishart(init=init, I=self.J, K=self.K, muU=self.muU, sigmaU=self.sigmaV)
        self.tau = initialise_tau_gamma(
            alpha=self.alpha, beta=self.beta, R=self.R, M=self.M, U=self.U, V=self.V)
        
        
    def run(self,iterations):
        """ Run the Gibbs sampler for the specified number of iterations. """
        self.all_U = numpy.zeros((iterations,self.I,self.K))  
        self.all_V = numpy.zeros((iterations,self.J,self.K))   
        self.all_muU = numpy.zeros((iterations,self.K))  
        self.all_muV = numpy.zeros((iterations,self.K))  
        self.all_sigmaU = numpy.zeros((iterations,self.K,self.K))  
        self.all_sigmaV = numpy.zeros((iterations,self.K,self.K))  
        self.all_tau = numpy.zeros(iterations) 
        self.all_times = []
        self.all_performances = { metric: [] for metric in METRICS } 
        
        time_start = time.time()
        for it in range(iterations):
            # Update the random variables
            self.muU, self.sigmaU = update_muU_sigmaU_gaussian_gaussian_wishart(
                mu0=self.mu0, beta0=self.beta0, v0=self.v0, W0=self.W0, U=self.U)
            self.U = update_U_gaussian_gaussian_wishart(
                muU=self.muU, sigmaU=self.sigmaU, R=self.R, M=self.M, V=self.V, tau=self.tau)
            
            self.muV, self.sigmaV = update_muV_sigmaV_gaussian_gaussian_wishart(
                mu0=self.mu0, beta0=self.beta0, v0=self.v0, W0=self.W0, V=self.V)
            self.V = update_V_gaussian_gaussian_wishart(
                muV=self.muV, sigmaV=self.sigmaV, R=self.R, M=self.M, U=self.U, tau=self.tau)
                 
            self.tau = update_tau_gaussian(
                alpha=self.alpha, beta=self.beta, R=self.R, M=self.M, U=self.U, V=self.V)
            
            # Store the draws
            self.all_U[it], self.all_V[it] = numpy.copy(self.U), numpy.copy(self.V)
            self.all_muU[it], self.all_sigmaU[it] = numpy.copy(self.muU), numpy.copy(self.sigmaU)
            self.all_muV[it], self.all_sigmaV[it] = numpy.copy(self.muV), numpy.copy(self.sigmaV)
            self.all_tau[it] = self.tau
            
            # Print the performance, store performance and time
            perf = self.predict_while_running()
            for metric in METRICS:
                self.all_performances[metric].append(perf[metric])
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)   
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
