"""
Class representing a Poisson distribution, allowing us to sample from it.
"""
from numpy.random import poisson

# Poisson draws
def poisson_draw(lamb):
    return poisson(lam=lamb)
        
'''
# Do 1000 draws and plot them
import matplotlib.pyplot as plt
lamb = 5.
s = [poisson_draw(lamb) for i in range(0,1000)] 
count, bins, ignored = plt.hist(s, 50, normed=True)
plt.show()
'''