"""
Class representing a Laplace distribution, allowing us to sample from it.
"""
from numpy.random import laplace

def laplace_draw(mu, lamb):
    return laplace(loc=mu, scale=lamb,size=None)
        
def laplace_mean(mu, lamb):
    return mu    
        
'''
# Do 1000 draws and plot them
import matplotlib.pyplot as plt
lamb = 2.
s = [laplace_draw(mu=0., lamb=lamb) for i in range(0,1000)] 
s2 = laplace(loc=0., scale=lamb, size=1000)
count, bins, ignored = plt.hist(s, 50, normed=True)
count, bins, ignored = plt.hist(s2, 50, normed=True)
plt.show()
'''