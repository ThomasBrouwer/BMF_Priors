"""
Bayesian Matrix Factorisation with Gaussian likelihood, Truncated Normal and 
Hierarchical priors.

Rij ~ N(Ui*Vj,tau^-1),                    tau ~ Gamma(alpha,beta),
Uik ~ TN(muU_ik,tauU_ik),                 Vjk ~ TN(muV_jk,tauV_jk),
muU_ik, tauU_ik ~ p(mu_mu, tau_mu, a, b), muV_jk, tauV_jk ~ p(mu_mu, tau_mu, a, b)

Random variables: U, V, muU, tauU, muV, tauV, tau.
Hyperparameters: alpha, beta, mu_mu, tau_mu, a, b.
"""

from bmf import BMF
from Gibbs.updates import update_tau_gaussian
from Gibbs.updates import update_U_gaussian_truncatednormal_hierarchical
from Gibbs.updates import update_V_gaussian_truncatednormal_hierarchical
from Gibbs.updates import update_muU_gaussian_truncatednormal_hierarchical
from Gibbs.updates import update_tauU_gaussian_truncatednormal_hierarchical
from Gibbs.updates import update_muV_gaussian_truncatednormal_hierarchical
from Gibbs.updates import update_tauV_gaussian_truncatednormal_hierarchical
from Gibbs.initialise import initialise_tau_gamma
from Gibbs.initialise import initialise_U_truncatednormal
from Gibbs.initialise import initialise_muU_tauU_hierarchical

import numpy
import time

METRICS = ['MSE', 'R^2', 'Rp']
OPTIONS_INIT = ['random', 'exp']
DEFAULT_HYPERPARAMETERS = {
    'alpha': 1.,
    'beta': 1.,
    'mu_mu': 0.1,
    'tau_mu': 0.1,
    'a': 1.,
    'b': 1.,
}

class BMF_Gaussian_TruncatedNormal_Hierarchical(BMF):
    def __init__(self,R,M,K,hyperparameters={}):
        """ Set up the class. """
        super(BMF_Gaussian_TruncatedNormal_Hierarchical, self).__init__(R, M, K)
        self.alpha =  hyperparameters.get('alpha',  DEFAULT_HYPERPARAMETERS['alpha'])
        self.beta =   hyperparameters.get('beta',   DEFAULT_HYPERPARAMETERS['beta'])   
        self.mu_mu =  hyperparameters.get('mu_mu',  DEFAULT_HYPERPARAMETERS['mu_mu']) 
        self.tau_mu = hyperparameters.get('tau_mu', DEFAULT_HYPERPARAMETERS['tau_mu'])   
        self.a =      hyperparameters.get('a',      DEFAULT_HYPERPARAMETERS['a']) 
        self.b =      hyperparameters.get('b',      DEFAULT_HYPERPARAMETERS['b'])  
        
        
    def initialise(self,init):
        """ Initialise the values of the random variables in this model. """
        assert init in OPTIONS_INIT, \
            "Unknown initialisation option: %s. Should be one of %s." % (init, OPTIONS_INIT)
        self.muU, self.tauU = initialise_muU_tauU_hierarchical(
            init=init, I=self.I, K=self.K, mu_mu=self.mu_mu, tau_mu=self.tau_mu, a=self.a, b=self.b)
        self.muV, self.tauV = initialise_muU_tauU_hierarchical(
            init=init, I=self.J, K=self.K, mu_mu=self.mu_mu, tau_mu=self.tau_mu, a=self.a, b=self.b)
        self.U = initialise_U_truncatednormal(
            init=init, I=self.I, K=self.K, mu=self.muU, tau=self.tauU)
        self.V = initialise_U_truncatednormal(
            init=init, I=self.J, K=self.K, mu=self.muV, tau=self.tauV)
        self.tau = initialise_tau_gamma(
            alpha=self.alpha, beta=self.beta, R=self.R, M=self.M, U=self.U, V=self.V)
        
        
    def run(self,iterations):
        """ Run the Gibbs sampler for the specified number of iterations. """
        self.all_U = numpy.zeros((iterations,self.I,self.K))  
        self.all_V = numpy.zeros((iterations,self.J,self.K))   
        self.all_muU = numpy.zeros((iterations,self.I,self.K))  
        self.all_muV = numpy.zeros((iterations,self.J,self.K))  
        self.all_tauU = numpy.zeros((iterations,self.I,self.K))  
        self.all_tauV = numpy.zeros((iterations,self.J,self.K))  
        self.all_tau = numpy.zeros(iterations) 
        self.all_times = []
        self.all_performances = { metric: [] for metric in METRICS } 
        
        time_start = time.time()
        for it in range(iterations):
            # Update the random variables
            self.muU = update_muU_gaussian_truncatednormal_hierarchical(
                mu_mu=self.mu_mu, tau_mu=self.tau_mu, U=self.U, tauU=self.tauU)
            self.tauU = update_tauU_gaussian_truncatednormal_hierarchical(
                a=self.a, b=self.b, U=self.U, muU=self.muU)
            self.U = update_U_gaussian_truncatednormal_hierarchical(
                muU=self.muU, tauU=self.tauU, R=self.R, M=self.M, U=self.U, V=self.V, tau=self.tau) 
            
            self.muV = update_muV_gaussian_truncatednormal_hierarchical(
                mu_mu=self.mu_mu, tau_mu=self.tau_mu, V=self.V, tauV=self.tauV)
            self.tauV = update_tauV_gaussian_truncatednormal_hierarchical(
                a=self.a, b=self.b, V=self.V, muV=self.muV)
            self.V = update_V_gaussian_truncatednormal_hierarchical(
                muV=self.muV, tauV=self.tauV, R=self.R, M=self.M, U=self.U, V=self.V, tau=self.tau) 
            
            self.tau = update_tau_gaussian(
                alpha=self.alpha, beta=self.beta, R=self.R, M=self.M, U=self.U, V=self.V)
            
            # Store the draws
            self.all_U[it], self.all_V[it] = numpy.copy(self.U), numpy.copy(self.V)
            self.all_muU[it], self.all_tauU[it] = numpy.copy(self.muU), numpy.copy(self.tauU)
            self.all_muV[it], self.all_tauV[it] = numpy.copy(self.muV), numpy.copy(self.tauV)
            self.all_U[it], self.all_V[it] = numpy.copy(self.U), numpy.copy(self.V)
            self.all_tau[it] = self.tau
            
            # Print the performance, store performance and time
            perf = self.predict_while_running()
            for metric in METRICS:
                self.all_performances[metric].append(perf[metric])
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)   
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
