"""
Methods for generating mask matrices, with 1 entries indicating observed and 0
indicating unobserved.
Provide methods for single mask matrices, and cross-validation folds.
"""

import numpy
import random
import itertools
from sklearn.cross_validation import StratifiedKFold


''' Helpers. '''
def compute_Ms(folds_M):
    ''' Take in the ten fold M's, and construct the masks M for the other nine folds. '''
    no_folds = len(folds_M)
    folds_M = [numpy.array(fold_M) for fold_M in folds_M]
    return [sum(folds_M[:fold]+folds_M[fold+1:]) for fold in range(0,no_folds)]


def nonzero_indices(M):
    ''' Return a list of indices of all nonzero indices in M. '''
    (I,J) = numpy.array(M).shape
    return [(i,j) for i,j in itertools.product(range(I),range(J)) if M[i,j]]


def check_empty_rows_columns(M):
    ''' Return True if all rows and columns have at least one observation. '''
    sums_columns = M.sum(axis=0)
    sums_rows = M.sum(axis=1)
                
    # Assert none of the rows or columns are entirely unknown values
    for i,c in enumerate(sums_rows):
        if c == 0:
            return False
    for j,c in enumerate(sums_columns):
        if c == 0:
            return False
    return True
    
def calc_inverse_M(M, M_combined=None):
    ''' Return the inverse of M. If M_combined is defined, make sure the 
        original plus inverse add up to be M_combined.'''
    return (M_combined if M_combined is not None else numpy.ones(M.shape)) - M
    

''' Generating methods. '''
def generate_M(I,J,fraction,M=None):
    ''' Generate a mask matrix M_train with :fraction missing entries. 
        If :M is defined, use only those 1-entries. '''
    if M is None:
        M = numpy.ones((I,J))
    indices = nonzero_indices(M)
    no_elements = len(indices)
    no_missing_total = I*J - no_elements
    assert no_missing_total < I*J*fraction, "Specified %s fraction missing, so %s entries missing, but there are already %s missing by default!" % \
        (fraction,I*J*fraction,no_missing_total)
    
    # Shuffle the observed entries, take the first (I*J)*(1-fraction) and mark those as observed
    M_train, M_test = numpy.zeros((I,J)), numpy.zeros((I,J))
    random.shuffle(indices)
    index_last_observed = int(I*J*(1-fraction))
    
    for i,j in indices[:index_last_observed]:
        M_train[i,j] = 1
    for i,j in indices[index_last_observed:]:
        M_test[i,j] = 1
    assert numpy.array_equal(M,M_train+M_test), "Tried splitting M into M_test and M_train but something went wrong."
    return M_train, M_test
    
    
def try_generate_M(I,J,fraction,attempts,M=None):
    ''' Try generate_M() :attempts times, making sure each row and column has 
        at least one observed entry. '''
    for i in range(attempts):
        M_train,M_test = generate_M(I=I,J=J,fraction=fraction,M=M)
        if check_empty_rows_columns(M_train):
            return M_train,M_test
    assert False, "Failed to generate folds for training and test data, %s attempts, fraction %s." % (attempts,fraction)


def compute_folds(I,J,no_folds,M=None):
    ''' Compute :no_folds cross-validation masks.
        Return a tuple (Ms_train, Ms_test), both a list of masks.
        If M is defined, split only the 1 entries into the folds. '''
    if M is None:
        M = numpy.ones((I,J))
        
    no_elements = M.sum()
    indices = nonzero_indices(M)
    
    random.shuffle(indices)
    split_places = [int(i*no_elements/no_folds) for i in range(0,no_folds+1)] #find the indices where the next fold start
    split_indices = [indices[split_places[i]:split_places[i+1]] for i in range(0,no_folds)] #split the indices list into the folds
    
    Ms_train, Ms_test = [], [] # list of the M's for the different folds
    for indices in split_indices:
        M_test = numpy.zeros(M.shape)
        for i,j in indices:
            M_test[i,j] = 1
        M_train = M - M_test
        Ms_train.append(M_train)
        Ms_test.append(M_test)
        
    return (Ms_train, Ms_test)
    

def compute_folds_attempts(I,J,no_folds,attempts,M=None):
    ''' Try generate_M() :attempts times, making sure each row and column of
        M_train has at least one observed entry. '''
    for i in range(attempts):
        Ms_train, Ms_test = compute_folds(I=I,J=J,no_folds=no_folds,M=M)
        success = True
        for M_train in Ms_train:
            if not check_empty_rows_columns(M_train):
                success = False
        if success:
            return Ms_train, Ms_test
    assert False, "Failed to generate folds for training and test data, %s attempts." % attempts
    
    
def compute_folds_stratify_rows(I,J,no_folds,M=None):
    ''' Run compute_folds() but make sure each fold has the same number of 
        entries of each row - i.e. we stratify the folds based on rows. 
        This could still create an empty column though. '''
    if M is None:
        M = numpy.ones((I,J))

    # Use skikit-learn stratified folds, using the row index as the labels
    indices = nonzero_indices(M)
    labels = [ind[0] for ind in indices]
    skf = StratifiedKFold(labels, no_folds, shuffle=True)
    Ms_train, Ms_test = [], []
    for _, test in skf:
        M_test = numpy.zeros(M.shape)
        for label_index in test:
            i,j = indices[label_index]
            M_test[i,j] = 1.
        M_train = M - M_test
        Ms_train.append(M_train), Ms_test.append(M_test)
    return Ms_train, Ms_test
    

def compute_folds_stratify_rows_attempts(I,J,no_folds,attempts,M=None):     
    ''' Try compute_folds_stratify_rows() :attempts times, making sure each row 
        and column of M_train has at least one observed entry. '''
    for i in range(attempts):
        Ms_train, Ms_test = compute_folds_stratify_rows(I=I,J=J,no_folds=no_folds,M=M)
        success = True
        for M_train in Ms_train:
            if not check_empty_rows_columns(M_train):
                success = False
        if success:
            return Ms_train, Ms_test
    assert False, "Failed to generate folds for training and test data, %s attempts." % attempts

        
def compute_folds_stratify_columns(I,J,no_folds,M=None):
    ''' Same as compute_folds_stratify_rows() but now stratify by column. '''
    Ms_train_T, Ms_test_T = compute_folds_stratify_rows(I=J, J=I, no_folds=no_folds, M=M.T if M is not None else None)
    Ms_train, Ms_test = [M_train.T for M_train in Ms_train_T], [M_test.T for M_test in Ms_test_T]
    return (Ms_train, Ms_test)
    
    
def compute_folds_stratify_columns_attempts(I,J,no_folds,attempts,M=None):     
    ''' Try compute_folds_stratify_columns() :attempts times, making sure each row 
        and column of M_train has at least one observed entry. '''
    for i in range(attempts):
        Ms_train, Ms_test = compute_folds_stratify_columns(I=I,J=J,no_folds=no_folds,M=M)
        success = True
        for M_train in Ms_train:
            if not check_empty_rows_columns(M_train):
                success = False
        if success:
            return Ms_train, Ms_test
    assert False, "Failed to generate folds for training and test data, %s attempts." % attempts


