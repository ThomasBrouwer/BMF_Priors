'''
Methods for loading in the Jester joke dataset. 

Rows are users, columns are jokes. Ratings are in the range [-10,10] originally
(99 means no rating), but we add 10 and cast them as integers.

SUMMARY: n_users, n_jokes, n_entries, fraction_obs
Jester:  73421,   100,     4136360,   0.5633756009860938
'''
import numpy
import itertools

folder_data = '/Users/thomasbrouwer/Documents/Projects/libraries/BMF_Priors/data/jester/' # '/home/tab43/Documents/Projects/libraries/BNMTF_ARD/data/methylation/' # 
file_jester_1 = folder_data+'jester-data-1.csv'
file_jester_2 = folder_data+'jester-data-2.csv'
file_jester_3 = folder_data+'jester-data-3.csv'

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
    
    
#'''    
R, M = load_jester_data()
R_int, M_int = load_jester_data_integer()
#'''