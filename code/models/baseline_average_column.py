"""
Simple baseline where we use the column average to make predictions.
"""

from bmf import BMF

import numpy

METRICS = ['MSE', 'R^2', 'Rp']

class ColumnAverage(BMF):
    def __init__(self,R,M,K,hyperparameters={}):
        """ Set up the class. """
        super(ColumnAverage, self).__init__(R, M, K)
        
                      
    def initialise(self,init):
        return
    
        
    def run(self,iterations):
        """ Compute the row averages of R. """
        column_averages = []
        for j in range(self.J):
            Rj_masked = [self.R[i,j] for i in range(self.I) if self.M[i,j]]
            average = numpy.average(Rj_masked)
            column_averages.append(average)
        self.column_averages = numpy.array(column_averages)
        
            
    """ Override the predict() method to use the row averages. """
    def predict(self,M_pred,burn_in,thinning):
        """ Use the row averages predict missing values. """
        R_pred = numpy.reshape(numpy.repeat(self.column_averages, self.I), (self.J, self.I)).T
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}