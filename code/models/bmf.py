"""
General class for the BMF models. 
Each specific model should extend this class, and implement methods:
    __init__() should set up the class.
    initialise(init) should initialise the random variables in the model.
    run(it) should run the Gibbs sampler for :iterations iterations.

USAGE
    BMF = bmf_gibbs(R, M, K, hyperparameters)
    BMF.initialise(init)
    BMF.run(it)
    performance = BMF.predict(M_pred, burn_in, thinning)
    U, V = BMF.approx_expectation_UV(burn_in, thinning)
where
    R is the matrix with observed values
    M is the mask matrix indicating observed values (1) and unobserved (0)
    K is the number of latent factors
    hyperparameters is a dictionary defining the priors over U, V, tau, etc. (or {} if using defaults)
    init defines the method of initialising the random variables ('random' or 'expectation')
    iterations is the number of iterations we run the method for
    burn_in is the number of iterations we skip before estimating the expectation
    thinning indicates which iterations we thin out (after burn_in)
    performance is a dictionary { 'MSE', 'R^2', 'Rp' }
    
The draw values are stored in all_U, all_V, all_tau, etc; performances in
all_performances; and timestamps in all_times.
"""

import numpy, math

class BMF(object):
    def __init__(self,R,M,K):
        """ Set up the class. """
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M,dtype=float)
        self.K = K
        
        assert len(self.R.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.R.shape)
        assert self.R.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.R.shape,self.M.shape)
            
        (self.I,self.J) = self.R.shape
        self.size_Omega = self.M.sum()
        self.check_empty_rows_columns()      
        
        
    def train(self,init,iterations):
        """ Initialise and run the model. """
        self.initialise(init=init)
        return self.run(iterations=iterations)

    def initialise(self,init):
        """ Initialise the values of the random variables in this model. """
        assert False, "Implement this method for your class!"
        
    def run(self,iterations):
        """ Run the Gibbs sampler for the specified number of iterations. """
        assert False, "Implement this method for your class!"
        
    
    def check_empty_rows_columns(self):
        """ Check if each row and column of M has at least 1 observed entry. """
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j


    def approx_expectation_UV(self,burn_in,thinning):
        """ Approximate the expectation of U and V (after burn_in and thinning), 
            returning a a tuple (U, V). """
        indices = range(burn_in,len(self.all_U),thinning)
        exp_U = numpy.array([self.all_U[i] for i in indices]).sum(axis=0) / float(len(indices))      
        exp_V = numpy.array([self.all_V[i] for i in indices]).sum(axis=0) / float(len(indices))  
        return (exp_U, exp_V)

    def predict(self,M_pred,burn_in,thinning):
        """ Compute the expectation of U and V, and use it to predict missing values. """
        U, V = self.approx_expectation_UV(burn_in,thinning)
        R_pred = numpy.dot(U,V.T)
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
    def predict_while_running(self):
        R_pred = numpy.dot(self.U,self.V.T)
        MSE = self.compute_MSE(self.M,self.R,R_pred)
        R2 = self.compute_R2(self.M,self.R,R_pred)    
        Rp = self.compute_Rp(self.M,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
    
    def compute_MSE(self,M,R,R_pred):
        """ Compute the MSE of entries in M, comparing R with R_pred. """
        return (M * (R-R_pred)**2).sum() / float(M.sum())
        
    def compute_R2(self,M,R,R_pred):
        """ Compute the R^2 of entries in M, comparing R with R_pred. """
        mean = (M*R).sum() / float(M.sum())
        SS_total = float((M*(R-mean)**2).sum())
        SS_res = float((M*(R-R_pred)**2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else numpy.inf
        
    def compute_Rp(self,M,R,R_pred):
        """ Compute the Rp of entries in M, comparing R with R_pred. """
        mean_real = (M*R).sum() / float(M.sum())
        mean_pred = (M*R_pred).sum() / float(M.sum())
        covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
        variance_real = (M*(R-mean_real)**2).sum()
        variance_pred = (M*(R_pred-mean_pred)**2).sum()
        return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred))

    def log_likelihood(self,expU,expV,exptau):
        """ Return the likelihood of the data given the trained model's parameters. """
        explogtau = math.log(exptau)
        return self.size_Omega / 2. * ( explogtau - math.log(2*math.pi) ) \
             - exptau / 2. * (self.M*( self.R - numpy.dot(expU,expV.T))**2).sum()