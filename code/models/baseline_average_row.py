"""
Simple baseline where we use the row average to make predictions.
"""

from bmf import BMF

import numpy

METRICS = ['MSE', 'R^2', 'Rp']


class RowAverage(BMF):
    def __init__(self,R,M,K,hyperparameters={}):
        """ Set up the class. """
        super(RowAverage, self).__init__(R, M, K)
        
                      
    def initialise(self,init):
        return
    
        
    def run(self,iterations):
        """ Compute the row averages of R. """
        row_averages = []
        for i in range(self.I):
            Ri_masked = [self.R[i,j] for j in range(self.J) if self.M[i,j]]
            average = numpy.average(Ri_masked)
            row_averages.append(average)
        self.row_averages = numpy.array(row_averages)
        
            
    """ Override the predict() method to use the row averages. """
    def predict(self,M_pred,burn_in,thinning):
        """ Use the row averages predict missing values. """
        R_pred = numpy.reshape(numpy.repeat(self.row_averages, self.J), (self.I, self.J))
        MSE = self.compute_MSE(M_pred,self.R,R_pred)
        R2 = self.compute_R2(M_pred,self.R,R_pred)    
        Rp = self.compute_Rp(M_pred,self.R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}