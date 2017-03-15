"""
General framework for performing nested cross validation for a matrix prediction
method. This can be used to find the performance of this method.

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
- P, the number of parallel threads
- parameter_search, a list of dictionaries from parameter names to values, 
    defining the space of our parameter search.
- train_config, the additional parameters to pass to the train function (e.g. no. of iterations).
    This should be a dictionary mapping parameter names to values 
- predict_config, the additional parameters to pass to the predict function 
    (e.g. burn_in and thinning). This should be a dictionary mapping parameter 
    names to values.
- file_performance, the location and name of the file in which we store the 
    overall performances of the nested cross-validations.
- files_nested_performances, a list of K locations+names of the files in which
    we store the performances of the parameter search cross-validation.

We split the dataset :R up into :K folds (considering only 1 entries in :M),
thus forming our :K training and test sets. Then for each we run the regular
cross-validation framework to find the best parameters on the training dataset. 
Then we train a model using these parameters, and evaluate it on the test set.
The performances are stored in :file_performance.
We use the row or column numbers to stratify the splitting of the entries into
masks. If we have more rows, we use column numbers; and vice versa.

We use the parallel matrix cross-validation module.

Methods:
- Constructor - simply takes in the arguments requires
- run - one argument (parallel), runs the cross validation and stores the 
    results in the file. If parallel=True, run the folds in parallel.
- find_best_parameters - takes in the name of the evaluation criterion (e.g. 
    'MSE'), and True if low is better (False if high is better), and returns 
    the best parameters based on that, in a tuple with all the performances.
    Also logs these findings to the file.
"""

from matrix_cross_validation import MatrixCrossValidation
from parallel_matrix_cross_validation import ParallelMatrixCrossValidation
from mask import compute_folds_stratify_rows_attempts
from mask import compute_folds_stratify_columns_attempts

import numpy

attempts_generate_M = 1000

class MatrixNestedCrossValidation:
    def __init__(self,method,R,M,K,P,parameter_search,train_config,predict_config,file_performance,files_nested_performances):
        self.method = method
        self.R = numpy.array(R,dtype=float)
        self.M = numpy.array(M)
        self.K = K
        self.P = P
        self.train_config = train_config
        self.predict_config = predict_config
        self.parameter_search = parameter_search
        self.files_nested_performances = files_nested_performances        
        
        self.fout = open(file_performance,'w')
        (self.I,self.J) = self.R.shape
        assert (self.R.shape == self.M.shape), "X and M are of different shapes: %s and %s respectively." % (self.R.shape,self.M.shape)
        
        self.all_performances = {}      # Performances across all folds - dictionary from evaluation criteria to a list of performances
        self.average_performances = {}  # Average performances across folds - dictionary from evaluation criteria to average performance
        
        
    def run(self, parallel=True):
        ''' Run the cross-validation. '''
        folds_method = compute_folds_stratify_rows_attempts if self.I < self.J else compute_folds_stratify_columns_attempts
        folds_training, folds_test = folds_method(I=self.I, J=self.J, no_folds=self.K, attempts=attempts_generate_M, M=self.M)
                
        for i,(train,test) in enumerate(zip(folds_training,folds_test)):
            print "Fold %s of nested cross-validation." % (i+1)            
            
            # Run the cross-validation
            crossval = ParallelMatrixCrossValidation(
                method=self.method,
                R=self.R,
                M=train,
                K=self.K,
                parameter_search=self.parameter_search,
                train_config=self.train_config,
                predict_config=self.predict_config,
                file_performance=self.files_nested_performances[i],
                P=self.P,
            ) if parallel else MatrixCrossValidation(
                method=self.method,
                R=self.R,
                M=train,
                K=self.K,
                parameter_search=self.parameter_search,
                train_config=self.train_config,
                predict_config=self.predict_config,
                file_performance=self.files_nested_performances[i],
            )
            crossval.run()
            
            try:
                (best_parameters,_) = crossval.find_best_parameters(evaluation_criterion='MSE',low_better=True)
                print "Best parameters for fold %s were %s." % (i+1,best_parameters)
            except KeyError:
                best_parameters = self.parameter_search[0]
                print "Found no performances, dataset too sparse? Use first values instead for fold %s, %s." % (i+1,best_parameters)
            
            # Train the model and test the performance on the test set
            performance_dict = self.run_model(train,test,best_parameters)
            self.store_performances(performance_dict)
            print "Finished fold %s, with performances %s." % (i+1,performance_dict)            
            
        self.log()
            
      
    def run_model(self,train,test,parameters):  
        ''' Initialises and runs the model, and returns the performance on the test set. '''
        model = self.method(self.R,train,**parameters)
        model.train(**self.train_config)
        return model.predict(test,**self.predict_config)
        
    
    def store_performances(self,performance_dict):
        ''' Store the performances we get back in a dictionary from criterion name to a list of performances. '''
        for name in performance_dict:
            if name in self.all_performances:
                self.all_performances[name].append(performance_dict[name])
            else:
                self.all_performances[name] = [performance_dict[name]]
              
    
    def compute_average_performances(self):
        ''' Compute the average performance of the given parameters, across the K folds. '''
        performances = self.all_performances     
        average_performances = { name:(sum(values)/float(len(values))) for (name,values) in performances.iteritems() }
        self.average_performances = average_performances
        
    
    def log(self):
        ''' Logs the performance on the test set for this fold. '''
        self.compute_average_performances()
        message = "Average performances: %s. \nAll performances: %s. \n" % (self.average_performances,self.all_performances)
        self.fout.write(message)
        self.fout.flush()