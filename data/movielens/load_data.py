'''
Methods for loading in the MovieLens datasets, which we construct from the raw
data. We could also store the numpy arrays and load those, but that is only
slightly faster (for MovieLens 1M, 1 minute vs 1:40 minutes).

Rows are users, columns are movies. Ratings are 0-5 (0 means no rating).

SUMMARY: n_users, n_movies, n_entries, fraction_obs
100K:    943      1682      100000     0.063046693642245313
1M:      6040     3706      1000209    0.044683625622312845
'''

import numpy

folder_data = '/Users/thomasbrouwer/Documents/Projects/libraries/BMF_Priors/data/movielens/' # '/home/tab43/Documents/Projects/libraries/BNMTF_ARD/data/movielens/' # 

folder_100K = folder_data+'100K/'
fin_100K = folder_100K+'u.data'

folder_1M = folder_data+'1M/'
fin_1M = folder_1M+'ratings.dat'

DELIM_100K = '\t'
DELIM_1M = '::'

MIN_NO_ENTRIES = 3

def construct_dataset_from_raw(fin, delim):
    ''' Return (R, M), which we construct from a file with one rating per line,
        of the form: user_id<delim>movie_id<delim>rating<delim>timestamp\n '''
    lines = [line.split('\n')[0].split(delim) for line in open(fin, 'r').readlines()]
    user_ids, movie_ids, ratings, timestamps = zip(*lines)
    user_ids = numpy.array(user_ids, dtype=int)
    movie_ids = numpy.array(movie_ids, dtype=int)
    ratings = numpy.array(ratings, dtype=int)
    
    # Construct the matrix
    unique_user_ids, unique_movie_ids = sorted(set(user_ids)), sorted(set(movie_ids))
    n_users, n_movies = len(unique_user_ids), len(unique_movie_ids)
    R, M = numpy.zeros((n_users, n_movies)), numpy.zeros((n_users, n_movies))
    for c, (user_id, movie_id, rating) in enumerate(zip(user_ids, movie_ids, ratings)):
        i, j = unique_user_ids.index(user_id), unique_movie_ids.index(movie_id)
        R[i,j], M[i,j] = rating, 1.
        if c % 10000 == 0:
            print "Constructing dataset... Done %s lines." % c
    print "Constructing dataset... Finished!"
    
    # Filter out any rows or columns with less than MIN_NO_ENTRIES  
    print "Before filtering, shape is (%s, %s)." % (R.shape)
    R, M = filter_rows(R, M, MIN_NO_ENTRIES) 
    print "After row filtering, shape is (%s, %s)." % (R.shape)
    R, M = filter_columns(R, M, MIN_NO_ENTRIES)
    print "After column filtering, shape is (%s, %s)." % (R.shape)
    R, M = filter_rows(R, M, MIN_NO_ENTRIES) 
    print "After second row filtering, shape is (%s, %s)." % (R.shape)
    return (R, M)

def filter_rows(R, M, min_no_entries):
    I, J = R.shape
    entries_per_row = M.sum(axis=1)
    indices_enough_entries = [i for i in range(I) if entries_per_row[i] >= min_no_entries]
    return R[indices_enough_entries,:], M[indices_enough_entries,:]

def filter_columns(R, M, min_no_entries):
    R_new, M_new = filter_rows(R.T, M.T, min_no_entries)
    return R_new.T, M_new.T

def load_movielens_100K():
    ''' Process and store files for MovieLens 100K. '''
    R, M = construct_dataset_from_raw(fin=fin_100K, delim=DELIM_100K)  
    return (R, M)
    
def load_movielens_1M():
    ''' Process and store files for MovieLens 1M. '''
    R, M = construct_dataset_from_raw(fin=fin_1M, delim=DELIM_1M)  
    return (R, M)


'''
R_100K, M_100K = load_movielens_100K()
R_1M, M_1M = load_movielens_1M()
'''