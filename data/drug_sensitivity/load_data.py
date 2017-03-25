'''
Methods for loading in the drug sensitivity datasets.

Rows are cell lines, drugs are columns.

Initially unobserved values are nan. We replace them by 0, and set those mask
entries to 0.

SUMMARY: n_cl, n_drugs, n_entries, fraction_obs
GDSC:    707,  139,     79262,     0.806549103009
CTRP:    887,  545,     387130,    0.800823309165
CCLE IC: 504,  24,      11670,     0.964781746032
CCLE EC: 504,  24,      7626,      0.630456349206

'''
import numpy
import itertools

folder_data = '/Users/thomasbrouwer/Documents/Projects/libraries/BMF_Priors/data/drug_sensitivity/' # '/home/tab43/Documents/Projects/libraries/BNMTF_ARD/data/drug_sensitivity/' # 

folder_gdsc_ic50 = folder_data+'GDSC/processed_all/'
file_gdsc_ic50 = folder_gdsc_ic50+'ic50.txt'

folder_ctrp_ec50 = folder_data+'CTRP/processed_all/'
file_ctrp_ec50 = folder_ctrp_ec50+'ec50.txt'

folder_ccle_ic50 = folder_data+'CCLE/processed_all/'
file_ccle_ic50 = folder_ccle_ic50+'ic50.txt'

folder_ccle_ec50 = folder_data+'CCLE/processed_all/'
file_ccle_ec50 = folder_ccle_ec50+'ec50.txt'

DELIM = '\t'
MIN_GDSC = 3 # minimum number of observed drugs per cell line

def load_data_create_mask(location):
    ''' Load in .txt file, and set mask entries for nan to 0. '''
    R = numpy.loadtxt(location, dtype=float, delimiter=DELIM)
    I,J = R.shape
    M = numpy.ones((I,J))
    for i,j in itertools.product(range(I),range(J)):
        if numpy.isnan(R[i,j]):
            R[i,j], M[i,j] = 0., 0.            
    return (R, M)

def load_gdsc_ic50(location=file_gdsc_ic50):
    ''' Return (R_gdsc, M_gdsc). Filter out cell lines with fewer than :MIN_GDSC observed entries. '''
    R, M = load_data_create_mask(location)
    sum_per_cell_line = M.sum(axis=1)
    indices_to_keep = [i for i in range(R.shape[0]) if sum_per_cell_line[i] >= MIN_GDSC]
    return R[indices_to_keep,:], M[indices_to_keep,:]

def load_ctrp_ec50(location=file_ctrp_ec50):
    ''' Return (R_ctrp, M_ctrp). '''
    return load_data_create_mask(location)

def load_ccle_ic50(location=file_ccle_ic50):
    ''' Return (R_ccle_ic50, M_ccle_ic50). '''
    return load_data_create_mask(location)

def load_ccle_ec50(location=file_ccle_ec50):
    ''' Return (R_ccle_ec50, M_ccle_ec50). '''
    return load_data_create_mask(location)


'''
R_gdsc, M_gdsc = load_gdsc_ic50()
R_ctrp, M_ctrp = load_ctrp_ec50()
R_ccle_ic, M_ccle_ic = load_ccle_ic50()
R_ccle_ec, M_ccle_ec = load_ccle_ec50()
'''
