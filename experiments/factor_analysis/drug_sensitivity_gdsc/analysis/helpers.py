"""
Methods for doing the analysis.
"""

import numpy


''' Compute the std or mean per column in U, V. '''
def std_per_factor(U):
    return numpy.array(U).std(axis=0).tolist()
    
def mean_per_factor(U):
    return numpy.array(U).mean(axis=0).tolist()
    
def std_per_factor_combined(U, V):
    return std_per_factor(numpy.append(U, V, axis=0))
    
def mean_per_factor_combined(U, V):
    return mean_per_factor(numpy.append(U, V, axis=0))
    
    
''' Reorder the matrices U, V, sorted by std or mean. '''
def sort_columns_by_std(U, V):
    ''' Reorder columns in U, V s.t. first column has highest std, last has lowest. '''
    U, V = numpy.array(U), numpy.array(V)
    K = U.shape[1]
    std_per_factor = std_per_factor_combined(U, V)
    index_and_std_per_factor = zip(range(K), std_per_factor)
    sorted_index_and_std = sorted(index_and_std_per_factor, key=lambda x:x[1], reverse=True)
    sorted_U, sorted_V = numpy.zeros(U.shape), numpy.zeros(V.shape)
    for k, (k_original, _) in enumerate(sorted_index_and_std):
        sorted_U[:,k], sorted_V[:,k] = U[:,k_original], V[:,k_original]
    return sorted_U, sorted_V
    
def sort_columns_by_mean(U, V):
    ''' Reorder columns in U, V s.t. first column has highest mean, last has lowest. '''
    U, V = numpy.array(U), numpy.array(V)
    K = U.shape[1]
    mean_per_factor = mean_per_factor_combined(U, V)
    index_and_mean_per_factor = zip(range(K), mean_per_factor)
    sorted_index_and_mean = sorted(index_and_mean_per_factor, key=lambda x:x[1], reverse=True)
    sorted_U, sorted_V = numpy.zeros(U.shape), numpy.zeros(V.shape)
    for k, (k_original, _) in enumerate(sorted_index_and_mean):
        sorted_U[:,k], sorted_V[:,k] = U[:,k_original], V[:,k_original]
    return sorted_U, sorted_V
    

''' Return the average mean and std across the repeats of U, V (after reordering by std or mean). '''
def average_mean_std(all_U, all_V, sort_by_std=False, use_absolute=False):
    ''' Return the average mean and std of the combined U and V matrices.
        If sort_by_std=True, use std to sort. Otherwise use mean.
        if use_absolute=True, make all values in U, V positive.
        (average_mean_U, average_std_U, average_mean_V, average_std_V). '''
    if use_absolute:
        all_U, all_V = numpy.abs(all_U).tolist(), numpy.abs(all_V).tolist()
    sort_method = sort_columns_by_std if sort_by_std else sort_columns_by_mean
    all_U_sorted, all_V_sorted = zip(*[sort_method(U,V) for U,V in zip(all_U, all_V)])
    all_mean_U, all_std_U = zip(*[(mean_per_factor(U),std_per_factor(U)) for U in all_U_sorted])
    all_mean_V, all_std_V = zip(*[(mean_per_factor(V),std_per_factor(V)) for V in all_V_sorted])
    average_mean_U, average_std_U = numpy.array(all_mean_U).mean(axis=0), numpy.array(all_std_U).mean(axis=0)
    average_mean_V, average_std_V = numpy.array(all_mean_V).mean(axis=0), numpy.array(all_std_V).mean(axis=0)
    return (average_mean_U, average_std_U, average_mean_V, average_std_V)
    