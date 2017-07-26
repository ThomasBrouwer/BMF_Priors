'''
Methods for loading in the Jester joke dataset. 

Rows are users, columns are jokes. Ratings are in the range [-10,10] originally
(99 means no rating), but we add 10 and cast them as integers.

SUMMARY: n_users, n_jokes, n_entries, fraction_obs
Jester:  73421,   100,     4136360,   0.5633756009860938
'''
import numpy
import itertools
import tables

folder_data = '/home/tab43/Documents/Projects/libraries/BMF_Priors/data/jester/' # '/Users/thomasbrouwer/Documents/Projects/libraries/BMF_Priors/data/jester/' # 
file_jester_1 = folder_data+'jester-data-1.csv'
file_jester_2 = folder_data+'jester-data-2.csv'
file_jester_3 = folder_data+'jester-data-3.csv'

file_binary_R = folder_data+'binary_R.h5'
file_binary_M = folder_data+'binary_M.h5'
file_binary_R_int = folder_data+'binary_R_int.h5'
file_binary_M_int = folder_data+'binary_M_int.h5'

DELIM = ','

def load_jester_data():
    ''' Return (R, M), with the 3 Jester files concatenated. Rows are users, columns are jokes. '''
    R1 = numpy.loadtxt(file_jester_1, delimiter=DELIM, dtype=str)[:,1:]
    R2 = numpy.loadtxt(file_jester_2, delimiter=DELIM, dtype=str)[:,1:]
    R3 = numpy.loadtxt(file_jester_3, delimiter=DELIM, dtype=str)[:,1:]
    R = numpy.concatenate((R1, R2, R3), axis=0)
    M = numpy.zeros(R.shape)
    I, J = R.shape    
    for i,j in itertools.product(range(I),range(J)):
        rating = float(R[i,j].split('"')[1])
        R[i,j] = rating
        M[i,j] = 1. if rating != 99 else 0.
    R = numpy.array(R, dtype=float)
    return (R, M)
    
def load_jester_data_integer():
    ''' Return (R, M) for the Jester data, with values in [0,1,..,20]. Rows are users, columns are jokes. '''
    (R, M) = load_jester_data()
    R += 10
    R = numpy.array(R, dtype=int)
    return (R, M)

def store_processed_jester():
    ''' Construct the datasets, and efficiently store them as binary HDF5 PyTables. '''
    R, M = load_jester_data()
    R_int, M_int = numpy.array(R+10, dtype=int), numpy.copy(M)
    
    def store_pytable(filename, dataset):
        # Method for storing each file as a binary HDF5 file
        f = tables.open_file(filename, 'w')
        atom = tables.Atom.from_dtype(dataset.dtype)
        filters = tables.Filters(complib='blosc', complevel=5)
        ds = f.create_carray(f.root, 'all_data', atom, dataset.shape, filters=filters)
        ds[:] = dataset
        f.close()

    store_pytable(file_binary_R, R)
    store_pytable(file_binary_M, M)
    store_pytable(file_binary_R_int, R_int)
    store_pytable(file_binary_M_int, M_int)
    
def load_processed_jester_data():
    ''' Load the Jester datasets, from the HDF5 PyTables binary files. '''
    hdf5_R = tables.open_file(file_binary_R)
    hdf5_M = tables.open_file(file_binary_M)
    R, M = hdf5_R.root.all_data[:], hdf5_M.root.all_data[:]
    return (R, M)

def load_processed_jester_data_integer():
    ''' Load the Jester (integer) datasets, from the HDF5 PyTables binary files. '''
    hdf5_R_int = tables.open_file(file_binary_R_int)
    hdf5_M_int = tables.open_file(file_binary_M_int)
    R_int, M_int = hdf5_R_int.root.all_data[:], hdf5_M_int.root.all_data[:]
    return (R_int, M_int)
    
    
'''    
import time
time0 = time.time()

print "Loading Jester."
R1, M1 = load_jester_data()
R1_int, M1_int = load_jester_data_integer()
time1 = time.time()
print "Took %s seconds." % (time1-time0)

print "Storing Jester."
store_processed_jester()
time2 = time.time()
print "Took %s seconds." % (time2-time1)

print "Loading processed Jester."
R2, M2 = load_processed_jester_data()
R2_int, M2_int = load_processed_jester_data_integer()
time3 = time.time()
print "Took %s seconds." % (time3-time2)

assert numpy.array_equal(R1, R2)
assert numpy.array_equal(M1, M2)
assert numpy.array_equal(R1_int, R2_int)
assert numpy.array_equal(M1_int, M2_int)
'''
