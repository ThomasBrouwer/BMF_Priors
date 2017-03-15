"""
Class representing a Half Normal distribution, allowing us to sample from it.

If X ~ N(0,sigma^2), then Y = |X| is distributed as a Half Normal.
https://en.wikipedia.org/wiki/Half-normal_distribution
"""
from scipy.stats import halfnorm

# Draw a value for x ~ HN(sigma)
def half_normal_draw(sigma):
    return halfnorm.rvs(loc=0,scale=sigma)
    
       
'''
# Example draw
sigma = 1.
print half_normal_draw(sigma)
'''