"""
Class representing a Dirichlet distribution, allowing us to sample from it.
"""
from numpy.random import dirichlet

# Dirichlet draws
def dirichlet_draw(alpha):
    return dirichlet(alpha=alpha)
    
def dirichlet_mean(alpha):
    return alpha / alpha.sum()
        
'''
# Example draws
alpha = [0.1, 0.1, 0.1, 0.1] # [100, 100, 100, 100] # 
s = dirichlet_draw(alpha)
assert s.sum() == 1.
print s
'''