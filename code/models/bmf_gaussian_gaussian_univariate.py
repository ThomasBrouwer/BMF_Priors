"""
Bayesian Matrix Factorisation with Gaussian likelihood and Gaussian priors
(univariate posterior).

Rij ~ N(Ui*Vj,tau^-1), tau ~ Gamma(alpha,beta), Ui ~ N(0,I/lamb), Vj ~ N(0,I/lamb)

Random variables: U, V, tau.
Hyperparameters: alpha, beta, lamb.
"""

from bmf_gaussian_gaussian_multivariate import BMF_Gaussian_Gaussian_multivariate
from Gibbs.updates import update_tau_gaussian
from Gibbs.updates import update_U_gaussian_gaussian_univariate
from Gibbs.updates import update_V_gaussian_gaussian_univariate

import numpy
import time

METRICS = ['MSE', 'R^2', 'Rp']
OPTIONS_INIT = ['random', 'exp']
DEFAULT_HYPERPARAMETERS = {
    'alpha': 1.,
    'beta': 1.,
    'lamb': 0.1,
}

class BMF_Gaussian_Gaussian_univariate(BMF_Gaussian_Gaussian_multivariate):
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
            self.U = update_U_gaussian_gaussian_univariate(
                lamb=self.lamb, R=self.R, M=self.M, U=self.U, V=self.V, tau=self.tau) 
            self.V = update_V_gaussian_gaussian_univariate(
                lamb=self.lamb, R=self.R, M=self.M, U=self.U, V=self.V, tau=self.tau)
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
