'''
This file contains the updates for the Gibbs samplers, computing the conditional
posterior parameter values.

Updates for U, V - format (Likelihood) Prior:
- (Gaussian) Gaussian
- (Gaussian) Gaussian + Wishart
- (Gaussian) Gaussian + Automatic Relevance Determination
- (Gaussian) Gaussian + Volume Prior
- (Gaussian) Exponential
- (Gaussian) Exponential + ARD
- (Gaussian) Truncated Normal
- (Gaussian) Truncated Normal + hierarchical
- (Gaussian) Half Normal
- (Poisson)  Gamma
- (Poisson)  Gamma + hierarchical
- (Poisson)  Dirichlet

Other updates:
- tau (noise) from Gamma [model: all with Gaussian likelihood]
- mu, Sigma from Normal - Inverse Wishart [model: Gaussian + Wishart]
- lambdak from Automatic Relevance Determination [model: Gaussian + Automatic Relevance Determination]
- lambdak Automatic Relevance Determination [model: Exponential + Automatic Relevance Determination]
- mu, tau from hierarchical Truncated Normal [model: Truncated Normal + hierarchical]
- zij from Multinomial [model: all with Poisson likelihood]
- hi from hierarchical Gamma [model: Gamma + hierarchical]
'''


import numpy


''' General Gaussian and Poisson models '''
def gaussian_tau_alpha_beta(alpha, beta, R, M, U, V):
    """ alpha_s and beta_s for tau (noise) in Gaussian models. """
    alpha_s = alpha + M.sum() / 2.
    squared_error = (M*(R-numpy.dot(U,V.T))**2).sum()
    beta_s = beta + squared_error / 2.
    return (alpha_s, beta_s)

def poisson_zij_n_p(Rij, Ui, Vj):
    """ n and p (vector) for zij with Mult(Rij,(Ui0Vj0,..,UiKVjK)) prior. """
    n = Rij
    p = Ui * Vj
    return (n, p)


''' (Gausian) Gaussian '''
def gaussian_gaussian_mu_sigma():
    #TODO:
    pass


''' (Gausian) Gaussian + Wishart '''
def gaussian_gaussian_wishart_mu_sigma():
    #TODO:
    pass

def gaussian_wishart_beta0_v0_mu0_W0():
    #TODO:
    pass


''' (Gausian) Gaussian + Automatic Relevance Determination '''
def gaussian_gaussian_ard_mu_sigma():
    #TODO:
    pass

def gaussian_ard_alpha_beta(alpha0, beta0, Uk, Vk):
    """ alpha_s and beta_s for lambdak with Gamma(alpha0,beta0) prior. """
    I, J = Uk.shape, Vk.shape
    alpha_s = alpha0 + I / 2. + J / 2.
    beta_s = beta0 + Uk.sum() / 2. + Vk.sum() / 2.
    return (alpha_s, beta_s)

''' (Gausian) Gaussian + Volume Prior '''
def gaussian_gaussian_vp_mu_sigma():
    #TODO:
    pass


''' (Gausian) Exponential '''
def gaussian_exponential_mu_tau(k, lamb, R, M, U, V, tau):
    """ mu and tau for Uik with Exp(lamb) prior. 
        We do updates per column of U (so Uk). """
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    assert U.shape[1] == V.shape[1]
    tauUk = tau * ( M * V[:,k]**2 ).sum(axis=1)
    muUk = 1. / tauUk * ( -lamb + tau * (M * ( (R-numpy.dot(U,V.T)+numpy.outer(U[:,k],V[:,k]))*V[:,k] )).sum(axis=1))
    assert tauUk.shape == muUk.shape    
    return (tauUk, muUk)


''' (Gausian) Exponential + Automatic Relevance Determination '''
def gaussian_exponential_ard_mu_tau(k, lambdak, R, M, U, V, tau):
    """ mu and tau for Uik with Exp(lambda_k) prior.
        We do updates per column of U (so Uk). """
    return gaussian_exponential_mu_tau(k=k, lamb=lambdak, R=R, M=M, U=U, V=V, tau=tau)

def exponential_ard_alpha_beta(alpha0, beta0, Uk, Vk):
    """ alpha_s and beta_s for lambdak with Gamma(alpha0,beta0) prior. """
    I, J = Uk.shape, Vk.shape
    alpha_s = alpha0 + I + J
    beta_s = beta0 + Uk.sum() + Vk.sum()
    return (alpha_s, beta_s)


''' (Gausian) Truncated Normal '''
def gaussian_tn_mu_tau(k, muU, tauU, R, M, U, V, tau):
    """ mu and tau for Uik with TN(muU,tauU) prior. 
        We do updates per column of U (so Uk). """
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    assert U.shape[1] == V.shape[1]
    tauUk = tauU + tau * ( M * V[:,k]**2 ).sum(axis=1)
    muUk = 1. / tauUk * ( muU * tauU + tau * (M * ( (R-numpy.dot(U,V.T)+numpy.outer(U[:,k],V[:,k]))*V[:,k] )).sum(axis=1))
    assert tauUk.shape == muUk.shape    
    return (tauUk, muUk)


''' (Gausian) Truncated Normal + hierarchical '''
def gaussian_tn_hierarchical_mu_tau(k, muUik, tauUik, R, M, U, V, tau):
    """ mu and tau for Uik with TN(muUik,tauUik) prior, with hierarchical prior 
        for muUik, tauUik. """
    return gaussian_tn_hierarchical_mu_tau(k=k, muU=muUik, tauU=tauUik, R=R, M=M, U=U, V=V, tau=tau)

def tn_hierarchical_mu_m_t(mu_mu, tau_mu, U, muU, tauU):
    """ m and t for mu^U_ik with hierarchical prior (hyperparams mu_mu, tau_mu).
        We compute the values for all i,k; so U, muU, tauU should all be a matrix. """
    assert U.shape == muU.shape and U.shape == tauU.shape
    t = tau_mu + tauU
    m = 1. / t * ( tauU * U + muU * tau_mu )
    assert m.shape == U.shape and t.shape == U.shape
    return (m, t)

def tn_hierarchical_tau_a_b(a, b, U, muU):
    """ a_s and b_s for tau^U_ik with hierarchical prior (hyperparams a, b).
        We compute the values for all i,k; so U, muU should both be a matrix. """
    assert U.shape == muU.shape
    a_s = a * numpy.ones(U.shape)
    b_s = b + ( U * muU )**2
    assert b_s
    return (a_s, b_s)


''' (Gausian) Half Normal '''
def gaussian_hn_mu_tau(k, sigma, R, M, U, V, tau):
    """ mu and tau for Uik with HN(sigma) prior. 
        We do updates per column of U (so Uk). """
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    assert U.shape[1] == V.shape[1]
    tauUk = 1. / sigma**2 + tau * ( M * V[:,k]**2 ).sum(axis=1)
    muUk = 1. / tauUk * ( tau * (M * ( (R-numpy.dot(U,V.T)+numpy.outer(U[:,k],V[:,k]))*V[:,k] )).sum(axis=1))
    assert tauUk.shape == muUk.shape    
    return (tauUk, muUk)


''' (Poisson) Gamma '''
def poisson_gamma_a_b(a, b, Mi, Vk, Zik):
    """ a_s and b_s for Uik with Gamma(a,b) prior. """
    a_s = a + (Mi * Zik).sum()
    b_s = b + (Mi * Vk).sum()
    return (a_s, b_s)
    
    
''' (Poisson) Gamma + hierarchical '''
def poisson_gamma_hierarchical_a_b(a, hiU, Mi, Vk, Zik):
    """ a_s and b_s for Uik with Gamma(a,h_i^U) prior, and h^U_i ~ Gamma(ap,ap/bp). """
    return poisson_gamma_a_b(a=a, b=hiU, Mi=Mi, Vk=Vk, Zik=Zik)

def gamma_hierarchical_hiU_a_b(ap, bp, a, Ui):
    """ a_s and b_s for h^U_i with Gamma(ap,ap/bp) prior, and Uik ~ Gamma(a,h_i^U). """
    K = Ui.shape
    a_s = ap + K * a
    b_s = bp + Ui.sum()
    return (a_s, b_s)


''' (Poisson) Dirichlet '''
def poisson_dirichlet_alpha(alpha, Mi, zi):
    """ alpha (vector) for Ui with Dir(alpha) prior in Poisson models. """
    assert Mi.shape == zi.shape[0] and alpha.shape == zi.shape[1]
    alpha_s = alpha + (Mi * zi.T).sum(axis=1)
    assert alpha_s.shape == alpha.shape
    return alpha_s