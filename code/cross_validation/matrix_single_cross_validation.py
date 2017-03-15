"""
General framework for performing a single cross validation for a matrix 
prediction method. We can use this to compute the cross validation performance
of a single method (without doing a parameter search).

We expect the following arguments:
- method, a class that performs matrix prediction, with the following functions:
    -> Constructor
       Taking in matrix X, mask matrix M, and other parameters P1,P2,...
    -> train
       Taking in a number of configuration parameters, and trains the model.
    -> predict
       Taking in the complete matrix X, a mask matrix M with 1 values where
       we with to evaluate the predictions, and returns a dictionary mapping
       performance measure names to their values.
       {'MSE','R2','Rp'} (Mean Square Error, R^2, Pearson correlation coefficient)
- R, the data matrix.
- M, a mask matrix with 1 values where entries in X are known, and 0 where they are not.
- K, the number of folds for cross-validation.
- parameters, the parameters to pass to the initialiser of the class (in addition
    to R and M). This should be a dictionary mapping parameter names to values.
- train_config, the parameters to pass to the train function (e.g. no. of iterations).
    This should be a dictionary mapping parameter names to values. 
- predict_config, the additional parameters to pass to the predict function 
    (e.g. burn_in and thinning). This should be a dictionary mapping parameter 
    names to values.
- file_performance, the location and name of the file in which we store the performances.

We split the dataset :R into :K folds (considering only 1 entries in :M), and 
thus form our :K training and test sets. Then for each we train the model using 
the parameters and training configuration :train_config. The performances are 
stored in :file_performance.
We use the row or column numbers to stratify the splitting of the entries into
masks. If we have more rows, we use column numbers; and vice versa.

Methods:
- Constructor - simply takes in the arguments requires
- run - no arguments, runs the cross validation and stores the results in the file
- find_best_parameters - takes in the name of the evaluation criterion (e.g. 
    'MSE'), and True if low is better (False if high is better), and returns 
    the best parameters based on that, in a tuple with all the performances.
    Also logs these findings to the file.
"""

from mask import compute_folds_stratify_rows_attempts
from mask import compute_folds_stratify_columns_attempts

import numpy

ATTEMPTS_GENERATE_M = 1000
METRICS = ['MSE', 'R^2', 'Rp']

class MatrixSingleCrossValidation:
    def __init__(self,method,R,M,K,parameters,train_config,predict_config,file_performance):
        self.method = method
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M)
        self.K = K
        self.parameters = parameters
        self.train_config = train_config
        self.predict_config = predict_config
        
        self.fout = open(file_performance,'w')
        (self.I,self.J) = self.R.shape
        assert (self.R.shape == self.M.shape), "X and M are of different shapes: %s and %s respectively." % (self.R.shape,self.M.shape)
        
        # Performances across all folds - dictionary from evaluation criteria to a list of performances        
        self.performances = {metric:[] for metric in METRICS}  
        
        
    def run(self):
        ''' Run the cross-validation. '''
        print "Running cross-validation framework."
        
        # Compute the mask matrices
        folds_method = compute_folds_stratify_rows_attempts if self.I < self.J else compute_folds_stratify_columns_attempts
        folds_training, folds_test = folds_method(I=self.I, J=self.J, no_folds=self.K, attempts=ATTEMPTS_GENERATE_M, M=self.M)
        
        # Run each fold and store the performances.
        for i,(train,test) in enumerate(zip(folds_training,folds_test)):
            print "Fold %s." % (i+1)
            performance_dict = self.run_model(train,test,self.parameters)
            self.log_performance(i+1,performance_dict)
        self.log_average_performance()            
            
            
    def run_model(self,train,test,parameters):
        ''' Initialises and runs the model, and returns the performance on the test set. '''
        model = self.method(self.R,train,**parameters)
        model.train(**self.train_config)
        return model.predict(test,**self.predict_config)
        
        
    def log_performance(self,fold,performance_dict):
        ''' Logs the best performances and parameters. '''
        message = "Performances fold %s: %s. \n" % (fold,performance_dict)
        self.fout.write(message)
        self.fout.flush()
        for metric in METRICS:
            self.performances[metric].append(performance_dict[metric])

    
    def log_average_performance(self):
        avr_performance = {metric: numpy.mean(self.performances[metric]) for metric in METRICS}
        message = "Average performance: %s. All performances: %s." % (avr_performance, self.performances)
        self.fout.write(message)
        self.fout.flush()