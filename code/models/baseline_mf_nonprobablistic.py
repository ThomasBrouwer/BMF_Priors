"""
Non-probabilistic non-negative matrix factorisation, as presented in
"Algorithms for Non-negative Matrix Factorization" (Lee and Seung, 2001).
We use this as a non-Bayesian performance baseline.

We change the notation to match ours: R = UV.T instead of V = WH.

The updates are then:
- Uik <- Uik * (sum_j Vjk * Rij / (Ui dot Vj)) / (sum_j Vjk)
- Vjk <- Vjk * (sum_i Uik * Rij / (Ui dot Vj)) / (sum_i Uik)
Or more efficiently using matrix operations:
- Uik <- Uik * (Mi dot [V.k * Ri / (Ui dot V.T)]) / (Mi dot V.k)
- Vjk <- Vjk * (M.j dot [U.k * R.j / (U dot Vj)]) / (M.j dot U.k)
And realising that elements in each column in U and V are independent:
- U.k <- U.k * sum(M * [V.k * (R / (U dot V.T))], axis=1) / sum(M dot V.k, axis=1)
- V.k <- V.k * sum(M * [U.k * (R / (U dot V.T))], axis=0) / sum(M dot U.k, axis=0)

We expect the following arguments:
- R, the matrix
- M, the mask matrix indicating observed values (1) and unobserved ones (0)
- K, the number of latent factors
    
Initialisation can be done by running the initialise(init) function. We initialise as follows:
- init_UV = 'ones'        -> U[i,k] = V[j,k] = 1
          = 'random'      -> U[i,k] ~ U(0,1), V[j,k] ~ U(0,1), 
          = 'exponential' -> U[i,k] ~ Exp(exponential_prior), V[j,k] ~ Exp(exponential_prior) 
  where exponential_prior is a hyperparameter (default 1).
"""

from bmf import BMF
from Gibbs.distributions.exponential import exponential_draw

import itertools
import numpy
import time

METRICS = ['MSE', 'R^2', 'Rp']
OPTIONS_INIT = ['ones', 'random', 'exp']
DEFAULT_HYPERPARAMETERS = {
    'exponential_prior': 1.
}

class MF_Nonprobabilistic(BMF):
    def __init__(self,R,M,K,hyperparameters={}):
        """ Set up the class. """
        super(MF_Nonprobabilistic, self).__init__(R, M, K)
        self.exponential_prior = hyperparameters.get('exponential_prior',  DEFAULT_HYPERPARAMETERS['exponential_prior'])
        
        # For computing the I-div it is easier if unknown values are 1's, not 0's, to avoid numerical issues
        self.R_excl_unknown = numpy.empty((self.I,self.J))
        for i,j in itertools.product(range(0,self.I),range(0,self.J)):
            self.R_excl_unknown[i,j] = self.R[i,j] if self.M[i,j] else 1.
                
                      
    def initialise(self,init):
        """ Initialise the values of the random variables in this model. """
        assert init in OPTIONS_INIT, \
            "Unknown initialisation option: %s. Should be one of %s." % (init, OPTIONS_INIT)
            
        if init == 'ones':
            self.U = numpy.ones((self.I,self.K))
            self.V = numpy.ones((self.J,self.K))
        elif init == 'random':
            self.U = numpy.random.rand(self.I,self.K)
            self.V = numpy.random.rand(self.J,self.K)
        elif init == 'exponential':
            self.U = numpy.empty((self.I,self.K))
            self.V = numpy.empty((self.J,self.K))
            for i,k in itertools.product(range(self.I),range(self.K)):        
                self.U[i,k] = exponential_draw(self.exponential_prior)
            for j,k in itertools.product(range(self.J),range(self.K)):
                self.V[j,k] = exponential_draw(self.exponential_prior)
    
        
    def run(self,iterations):
        """ Run the Gibbs sampler for the specified number of iterations. """
        assert hasattr(self,'U') and hasattr(self,'V'), "U and V have not been initialised - please run initialise() first."        
        self.all_U = numpy.zeros((iterations,self.I,self.K))  
        self.all_V = numpy.zeros((iterations,self.J,self.K))
        self.all_times = []
        self.all_performances = { metric: [] for metric in METRICS } 
            
        time_start = time.time()
        for it in range(1,iterations+1):
            # Update the matrices U, V
            for k in range(self.K):
                self.update_U(k)
            for k in range(self.K):
                self.update_V(k)
            
            # Store the values
            self.all_U[it], self.all_V[it] = numpy.copy(self.U), numpy.copy(self.V)
            
            # Print the performance, store performance and time
            perf = self.predict_while_running()
            for metric in METRICS:
                self.all_performances[metric].append(perf[metric])
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)   
            print "Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (it+1,perf['MSE'],perf['R^2'],perf['Rp'])
            

    ''' Updates for U and V. '''
    def update_U(self,k):
        ''' Update values for U. '''
        self.U[:,k] = self.U[:,k] * (self.M * (self.V[:,k] * ( self.R / numpy.dot(self.U,self.V.T) ) )).sum(axis=1) / (self.M * self.V[:,k]).sum(axis=1)
        
    def update_V(self,k):
        ''' Update values for V. '''
        self.V[:,k] = self.V[:,k] * ( (self.U[:,k] * ( self.R / numpy.dot(self.U,self.V.T) ).T ).T * self.M ).sum(axis=0) / (self.U[:,k] * self.M.T).T.sum(axis=0)
        
        
    """ Override the predict() method to simply use U and V directly. """
    def predict(self,M_pred,burn_in,thinning):
        """ Use U and V to predict missing values. """
        R_pred = numpy.dot(self.U, self.V.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}