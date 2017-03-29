"""
Class representing a Multinomial distribution, allowing us to sample from it.
"""
from numpy.random import multinomial
import numpy

# Multinomial draws
def multinomial_draw(n,p):
    return multinomial(n=n,pvals=p)
        
def multinomial_mean(n,p):
    return n*numpy.array(p)        
        
'''
# Example draws
n, p = 10000, [0.1, 0.2, 0.3, 0.4]
s = multinomial_draw(n,p)
print s
'''