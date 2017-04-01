'''
This file contains the parameter value updates for the Gibbs samplers, 
computing the conditional posterior parameter values.

Parameters for U, V - format (Likelihood) Prior:
- (Gaussian) Gaussian (univariate posterior)
- (Gaussian) Gaussian (multivariate posterior)
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

Other parameters for:
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

def poisson_Zij_n_p(Rij, Ui, Vj):
    """ n and p (vector) for Zij with Mult(Rij,(Ui0Vj0,..,UiKVjK)) prior. """
    n = Rij
    p = Ui * Vj
    p /= p.sum()
    return (n, p)


''' (Gausian) Gaussian (univariate posterior). '''
def gaussian_gaussian_mu_tau(k, lamb, R, M, U, V, tau):
    """ muUk and tauUk (vectors) for Uk with N(0,I/lamb) prior (I=identity matrix). """
    I, J, K = R.shape[0], R.shape[1], U.shape[1]
    assert R.shape == M.shape and V.shape == (J,K) and U.shape[0] == I
    tauUk = lamb + tau * ( M * V[:,k]**2 ).sum(axis=1)
    muUk = tau * ( M * ( ( R - numpy.dot(U,V.T) + numpy.outer(U[:,k],V[:,k])) * V[:,k] ) ).sum(axis=1)
    muUk /= tauUk   
    assert muUk.shape == (I,) and tauUk.shape == (I,)
    return (muUk, tauUk)


''' (Gausian) Gaussian (multivariate posterior). '''
def gaussian_gaussian_mu_sigma(lamb, Ri, Mi, V, tau):
    """ mu and sigma for Ui with N(0,I/lamb) prior (I=identity matrix). """
    assert Ri.shape == Mi.shape and Ri.shape[0] == V.shape[0]
    K = V.shape[1]
    V_masked = (Mi * V.T).T # zero rows when j not in Mi
    precision = lamb * numpy.eye(K) + tau * ( numpy.dot(V_masked.T,V_masked) )
    sigma = numpy.linalg.inv(precision)
    Ri_masked = Mi * Ri # zero entries when j not in Mi
    mu = numpy.dot(sigma, tau * numpy.dot(Ri_masked, V))
    assert mu.shape[0] == V.shape[1] and sigma.shape == (V.shape[1], V.shape[1])
    return (mu, sigma)


''' (Gausian) Gaussian + Wishart '''
def gaussian_gaussian_wishart_mu_sigma(muU, sigmaU_inv, Ri, Mi, V, tau):
    """ mu and sigma for Ui with N(muU,sigmaU) prior. """
    assert Ri.shape == Mi.shape and Ri.shape[0] == V.shape[0]
    V_masked = (Mi * V.T).T # zero rows when j not in Mi
    precision = sigmaU_inv + tau * ( numpy.dot(V_masked.T,V_masked) )
    sigma = numpy.linalg.inv(precision)
    Ri_masked = Mi * Ri # zero entries when j not in Mi
    mu = numpy.dot(sigma, numpy.dot(sigmaU_inv, muU) + tau * numpy.dot(Ri_masked, V))
    assert mu.shape[0] == V.shape[1] and sigma.shape == (V.shape[1], V.shape[1])
    return (mu, sigma)

def gaussian_wishart_beta0_v0_mu0_W0(beta0, v0, mu0, W0, U):
    """ beta0_s, v0_s, mu0_s, W0_s for muU, sigmaU with NIW(beta0,v0,mu0,W0) prior. """
    I, K = U.shape
    assert mu0.shape == (K,) and W0.shape == (K, K)
    beta0_s = beta0 + I
    v0_s = v0 + I
    U_bar = U.sum(axis=0) / float(I) # vector giving average per column of U
    S_bar = numpy.dot(U.T, U) / float(I) # matrix giving covariance of columns of U
    mu0_s = ( beta0 * mu0 + I * U_bar) / ( beta0 + I )
    W0_s = W0 + I * S_bar + (beta0*I)/float(beta0+I) * numpy.outer(mu0-U_bar, mu0-U_bar)
    assert mu0_s.shape == (K,) and W0_s.shape == (K, K)
    return (beta0_s, v0_s, mu0_s, W0_s)


''' (Gausian) Gaussian + Automatic Relevance Determination '''
def gaussian_gaussian_ard_mu_sigma(lamb, Ri, Mi, V, tau):
    """ mu and sigma for Ui with N(0,diag(1/lamb)) prior. lamb is a vector. """
    assert Ri.shape == Mi.shape and Ri.shape[0] == V.shape[0] and lamb.shape[0] == V.shape[1]
    V_masked = (Mi * V.T).T # zero rows when j not in Mi
    precision = numpy.diag(lamb) + tau * ( numpy.dot(V_masked.T,V_masked) )
    sigma = numpy.linalg.inv(precision)
    Ri_masked = Mi * Ri # zero entries when j not in Mi
    mu = numpy.dot(sigma, tau * numpy.dot(Ri_masked, V))
    assert mu.shape[0] == V.shape[1] and sigma.shape == (V.shape[1], V.shape[1])
    return (mu, sigma)

def gaussian_ard_alpha_beta(alpha0, beta0, Uk, Vk):
    """ alpha_s and beta_s for lambdak with Gamma(alpha0,beta0) prior. """
    I, J = Uk.shape[0], Vk.shape[0]
    alpha_s = alpha0 + I / 2. + J / 2.
    beta_s = beta0 + (Uk**2).sum() / 2. + (Vk**2).sum() / 2.
    return (alpha_s, beta_s)


''' (Gausian) Gaussian + Volume Prior '''
def adjugate_matrix(matrix):
    """ adj(matrix) = det(matrix) matrix^-1 """
    return numpy.linalg.det(matrix) * numpy.linalg.inv(matrix)

def gaussian_gaussian_volumeprior_mu_sigma(i, k, gamma, Ri, Mi, U, V, tau):
    """ muUik and tauUik for Uik with Volume Prior, exp{-gamma det(U.T U)}. """
    I, K, J = U.shape[0], U.shape[1], Ri.shape[0]
    assert Mi.shape == (J,) and V.shape == (J, K)
    U_i_ktilde = numpy.append(U[i,:k],U[i,k+1:]) # vector Ui excl entry k
    U_itilde_k = numpy.append(U[:i,k],U[i+1:,k]) # vector Uk excl entry i
    U_ktilde = numpy.append(U[:,:k],U[:,k+1:],axis=1) # matrix U excl column k
    U_itilde_ktilde = numpy.append(U_ktilde[:i,:],U_ktilde[i+1:,:],axis=0) # matrix U excl row i and column k
    assert U_i_ktilde.shape == (K-1,) and U_itilde_k.shape == (I-1,)
    assert U_ktilde.shape == (I,K-1) and U_itilde_ktilde.shape == (I-1,K-1)
    cov_U_ktilde = numpy.dot(U_ktilde.T, U_ktilde)
    D_ktilde_ktilde = numpy.linalg.det(cov_U_ktilde)
    A_ktilde_ktilde = adjugate_matrix(cov_U_ktilde)
    assert cov_U_ktilde.shape == (K-1,K-1) and A_ktilde_ktilde.shape == (K-1,K-1)
    tauUik = tau*(Mi*V[:,k]**2).sum() + gamma * (D_ktilde_ktilde - numpy.dot(numpy.dot(U_i_ktilde,A_ktilde_ktilde),U_i_ktilde))
    #muUik = 1./tauUik * (
    #    tau * (Mi *((Ri-numpy.dot(U[i,:],V.T)+U[i,k]*V[:,k])*V[:,k])).sum() + 
    #    gamma * numpy.dot(numpy.dot(U_i_ktilde,A_ktilde_ktilde), numpy.dot(U_itilde_ktilde.T,U_itilde_k))
    #)
    V_ktilde = numpy.append(V[:,:k],V[:,k+1:],axis=1)
    muUik = 1./tauUik * (
        tau * (numpy.dot(Mi*Ri, V[:,k]) - numpy.dot(numpy.dot(U_i_ktilde, V_ktilde.T), Mi*V[:,k])) + 
        gamma * numpy.dot(numpy.dot(U_i_ktilde,A_ktilde_ktilde), numpy.dot(U_itilde_ktilde.T,U_itilde_k))
    )
    return (muUik, tauUik)


''' (Gausian) Exponential '''
def gaussian_exponential_mu_tau(k, lamb, R, M, U, V, tau):
    """ muUik and tauUik for Uik with Exp(lamb) prior. 
        We do updates per column of U (so Uk). """
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    assert U.shape[1] == V.shape[1]
    tauUk = tau * ( M * V[:,k]**2 ).sum(axis=1)
    #muUk = 1. / tauUk * ( -lamb + tau * (M * ( (R-numpy.dot(U,V.T)+numpy.outer(U[:,k],V[:,k]))*V[:,k] )).sum(axis=1))
    V_ktilde = numpy.append(V[:,:k],V[:,k+1:],axis=1)
    U_ktilde = numpy.append(U[:,:k],U[:,k+1:],axis=1)
    muUk = 1. / tauUk * ( -lamb + tau * (
        numpy.dot(M*R, V[:,k]) - numpy.dot(M*numpy.dot(U_ktilde, V_ktilde.T), V[:,k])) )
    assert tauUk.shape == muUk.shape
    return (muUk, tauUk)


''' (Gausian) Exponential + Automatic Relevance Determination '''
def gaussian_exponential_ard_mu_tau(k, lambdak, R, M, U, V, tau):
    """ mu and tau for Uik with Exp(lambda_k) prior.
        We do updates per column of U (so Uk). """
    return gaussian_exponential_mu_tau(k=k, lamb=lambdak, R=R, M=M, U=U, V=V, tau=tau)

def exponential_ard_alpha_beta(alpha0, beta0, Uk, Vk):
    """ alpha_s and beta_s for lambdak with Gamma(alpha0,beta0) prior. """
    I, J = Uk.shape[0], Vk.shape[0]
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
    #muUk = 1. / tauUk * ( muU * tauU + tau * (M * ( (R-numpy.dot(U,V.T)+numpy.outer(U[:,k],V[:,k]))*V[:,k] )).sum(axis=1))
    V_ktilde = numpy.append(V[:,:k],V[:,k+1:],axis=1)
    U_ktilde = numpy.append(U[:,:k],U[:,k+1:],axis=1)
    muUk = 1. / tauUk * ( muU * tauU + tau * (
        numpy.dot(M*R, V[:,k]) - numpy.dot(M*numpy.dot(U_ktilde, V_ktilde.T), V[:,k])) )
    assert tauUk.shape == muUk.shape    
    return (muUk, tauUk)


''' (Gausian) Truncated Normal + hierarchical '''
def gaussian_tn_hierarchical_mu_tau(k, muUk, tauUk, R, M, U, V, tau):
    """ mu and tau for Uik with TN(muUik,tauUik) prior, with hierarchical prior 
        for muUik, tauUik. We do updates per column of U (so Uk). """
    return gaussian_tn_mu_tau(k=k, muU=muUk, tauU=tauUk, R=R, M=M, U=U, V=V, tau=tau)

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
    a_s = a * numpy.ones(U.shape) + 0.5
    b_s = b + ( U - muU )**2 / 2.
    #print a_s, b_s, U, muU, a, b
    return (a_s, b_s)


''' (Gausian) Half Normal '''
def gaussian_hn_mu_tau(k, sigma, R, M, U, V, tau):
    """ mu and tau for Uik with HN(sigma) prior. 
        We do updates per column of U (so Uk). """
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    assert U.shape[1] == V.shape[1]
    tauUk = 1. / sigma**2 + tau * ( M * V[:,k]**2 ).sum(axis=1)
    #muUk = 1. / tauUk * ( tau * (M * ( (R-numpy.dot(U,V.T)+numpy.outer(U[:,k],V[:,k]))*V[:,k] )).sum(axis=1))
    V_ktilde = numpy.append(V[:,:k],V[:,k+1:],axis=1)
    U_ktilde = numpy.append(U[:,:k],U[:,k+1:],axis=1)
    muUk = 1. / tauUk * ( tau * (
        numpy.dot(M*R, V[:,k]) - numpy.dot(M*numpy.dot(U_ktilde, V_ktilde.T), V[:,k])) )
    assert tauUk.shape == muUk.shape   
    return (muUk, tauUk)


''' (Poisson) Gamma '''
def poisson_gamma_a_b(a, b, Mi, Vk, Zik):
    """ a_s and b_s for Uik with Gamma(a,b) prior. """
    a_s = a + (Mi * Zik).sum()
    b_s = b + (Mi * Vk).sum()
    return (a_s, b_s)
    
    
''' (Poisson) Gamma + hierarchical '''
def poisson_gamma_hierarchical_a_b(a, hUi, Mi, Vk, Zik):
    """ a_s and b_s for Uik with Gamma(a,h^U_i) prior, and h^U_i ~ Gamma(ap,ap/bp). """
    return poisson_gamma_a_b(a=a, b=hUi, Mi=Mi, Vk=Vk, Zik=Zik)

def gamma_hierarchical_hUi_a_b(ap, bp, a, Ui):
    """ a_s and b_s for h^U_i with Gamma(ap,ap/bp) prior, and Uik ~ Gamma(a,h_i^U). """
    K = Ui.shape[0]
    a_s = ap + K * a
    b_s = bp + Ui.sum()
    return (a_s, b_s)


''' (Poisson) Dirichlet '''
def poisson_dirichlet_alpha(alpha, Mi, Zi):
    """ alpha (vector) for Ui with Dir(alpha) prior in Poisson models. """
    assert Mi.shape[0] == Zi.shape[0] and alpha.shape[0] == Zi.shape[1]
    alpha_s = alpha + (Mi * Zi.T).sum(axis=1)
    assert alpha_s.shape == alpha.shape
    return alpha_s