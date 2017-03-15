"""
Class representing a multivariate normal distribution, allowing us to sample from it.
"""
from numpy.random import multivariate_normal
import numpy

def MN_draw(mu,precision=None,sigma=None):
    assert precision is not None or sigma is not None, "Need either Sigma or Precision."
    if sigma is None:
        sigma = numpy.linalg.inv(precision)
    return multivariate_normal(mean=mu,cov=sigma,size=None)