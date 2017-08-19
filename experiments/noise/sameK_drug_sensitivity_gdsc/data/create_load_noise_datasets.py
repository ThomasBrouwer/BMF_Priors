'''
Methods for creating the noisy versions of the drug sensitivity datasets, and
then loading them in.
'''

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BMF_Priors.code.models.Gibbs.distributions.normal import normal_draw
from BMF_Priors.data.drug_sensitivity.load_data import load_gdsc_ic50_integer

import itertools
import numpy


NOISE_TO_SIGNAL_RATIOS = [0., 0.1, 0.2, 0.5, 1., 2., 5., 10.]
FOLDER_FILES = project_location+'BMF_Priors/experiments/noise/drug_sensitivity_gdsc/data/'
FILE_NAMES_R = [FOLDER_FILES+'R_%s.txt' % int(100*NSR) for NSR in NOISE_TO_SIGNAL_RATIOS]
FILE_NAME_M = FOLDER_FILES+'M.txt'


def create_noise_datasets():
    ''' For each NSR, create a version of R with that level of noise. '''
    R, M = load_gdsc_ic50_integer()
    numpy.savetxt(FILE_NAME_M, M)
    for NSR, filename in zip(NOISE_TO_SIGNAL_RATIOS, FILE_NAMES_R):
        R_noise = add_noise(R=R, NSR=NSR)
        numpy.savetxt(filename, R_noise)
    
    
def load_noise_datasets():
    ''' Return (noise_to_signal_ratios, [R, .., R], M). '''
    Rs_noise = [numpy.loadtxt(filename) for filename in FILE_NAMES_R]
    M = numpy.loadtxt(FILE_NAME_M)
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