'''
This file contains methods for initialising random variables, such as U, V, tau, etc.
We either use the expectation or random draws of the prior distributions.
'''

from updates import update_tau_gaussian
from distributions.normal import normal_draw

import itertools
import numpy


def initialise_tau_gamma(alpha, beta, R, M, U, V):
    """ Initialise tau using the model updates. """
    return update_tau_gaussian(alpha=alpha, beta=beta, R=R, M=M, U=U, V=V)

def initialise_Z_multinomial():
    #TODO:
    pass

def initialise_U_gaussian(init, I, K, lamb):
    """ Initialise U randomly, with prior Ui ~ N(0,I/lamb). """
    U = numpy.zeros((I,K))
    for i,k in itertools.product(range(I),range(K)):
        U[i,k] = ( normal_draw(mu=0, tau=lamb) if init == 'random' else 0. )
    return U

def initialise_U_exponential():
    #TODO:
    pass

def initialise_U_truncated_normal():
    #TODO:
    pass

def initialise_U_half_normal():
    #TODO:
    pass

#TODO: many more