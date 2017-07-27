'''
Methods for creating the noisy versions of the MovieLens 100K datasets, and
then loading them in.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.Gibbs.distributions.normal import normal_draw
from BMF_Priors.data.movielens.load_data import load_processed_movielens_100K

import itertools
import numpy
import tables


NOISE_TO_SIGNAL_RATIOS = [0., 0.1, 0.2, 0.5, 1., 2., 5., 10.]
FOLDER_FILES = project_location+'BMF_Priors/experiments/noise/movielens_100K/data/'
FILE_NAMES_R = [FOLDER_FILES+'R_%s.h5' % int(100*NSR) for NSR in NOISE_TO_SIGNAL_RATIOS]
FILE_NAME_M = FOLDER_FILES+'M.h5'


def store_pytable(filename, dataset):
    ''' Method for storing each file as a binary HDF5 file. '''
    f = tables.open_file(filename, 'w')
    atom = tables.Atom.from_dtype(dataset.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = f.create_carray(f.root, 'all_data', atom, dataset.shape, filters=filters)
    ds[:] = dataset
    f.close()
    

def create_noise_datasets():
    ''' For each NSR, create a version of R with that level of noise. '''
    R, M = load_processed_movielens_100K()
    store_pytable(FILE_NAME_M, M)
    for NSR, filename in zip(NOISE_TO_SIGNAL_RATIOS, FILE_NAMES_R):
        R_noise = add_noise(R=R, NSR=NSR)
        store_pytable(filename, R_noise)
        
    
def load_noise_datasets():
    ''' Return (noise_to_signal_ratios, [R, .., R], M). '''
    Rs_noise = [tables.open_file(filename).root.all_data[:] for filename in FILE_NAMES_R]
    M = tables.open_file(FILE_NAME_M).root.all_data[:]
    return (NOISE_TO_SIGNAL_RATIOS, Rs_noise, M)


def add_noise(R, NSR):
    ''' Method for adding noise to a dataset, of the specified noise-to-signal 
        ratio. Also cast values to integers, and make nonnegative values zero. '''
    variance_signal = R.var()
    tau = 1. / (variance_signal * NSR)
    print "Noise: %s%%. Variance in dataset is %s. Adding noise with variance %s." % (100.*NSR,variance_signal,1./tau)
    
    if numpy.isinf(tau):
        return numpy.copy(R)
    (I,J) = R.shape
    R_noise = numpy.zeros((I,J))
    for i,j in itertools.product(range(I),range(J)):
        new_Rij = normal_draw(R[i,j],tau)
        R_noise[i,j] = max(0, int(new_Rij))
    
    print "New variance: %s." % (R_noise.var())
    return R_noise


if __name__ == '__main__':
    create_noise_datasets()