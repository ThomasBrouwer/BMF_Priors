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

NOTE:
We use minpy (https://github.com/dmlc/minpy) as a library for speeding up numpy
operations, using GPU operations. The library is pretty flawed though, and does
not support a lot of operations. We therefore only use it for dot and outer 
products. Immediately afterwards we have to cast the result back to a numpy 
array.
If you do not want to use minpy, use "import numpy as minpy".
'''


import numpy 
import minpy.numpy as minpy # import numpy as minpy # 

def cast_to_numpy(A):
    ''' If we use a minpy operation (dot or outer), cast back to pure numpy array. '''
    return A.asnumpy() if numpy != minpy else A

def minpy_dot(A1, A2):
    ''' Do minpy.dot(), but cast back to numpy array. '''
    return cast_to_numpy(minpy.dot(A1, A2))

def minpy_outer(A1, A2):
    ''' Do minpy.outer(), but cast back to numpy array. '''
    return cast_to_numpy(minpy.outer(A1, A2))


''' General Gaussian and Poisson models '''
def gaussian_tau_alpha_beta(alpha, beta, R, M, U, V):
    """ alpha_s and beta_s for tau (noise) in Gaussian models. """
    alpha_s = alpha + numpy.sum(M) / 2.
    squared_error = numpy.sum(M*(R-minpy_dot(U,V.T))**2)
    beta_s = beta + squared_error / 2.
    return (alpha_s, beta_s)

def poisson_Zij_n_p(Rij, Ui, Vj):
    """ n and p (vector) for Zij with Mult(Rij,(Ui0Vj0,..,UiKVjK)) prior. """
    n = Rij
    p = Ui * Vj
    p /= numpy.sum(p)
    return (n, p)

#def poisson_Z_n_p(R, U, V):
#    """ n (IxJ) and p (IxJxK) for all Zij with Mult(Rij,(Ui0Vj0,..,UiKVjK)) prior. """
#    I, J, K = R.shape[0], R.shape[1], U.shape[1]
#    n = R
#    U_extended = numpy.repeat(U[:,numpy.newaxis,:], J, axis=1)
#    V_extended = numpy.repeat(V[numpy.newaxis,:,:], I, axis=0)
#    p = U_extended * V_extended
#    p_sum = numpy.repeat(numpy.sum(p,axis=2)[:,:,numpy.newaxis], K, axis=2)
#    p /= p_sum
#    return (n, p)
    
def poisson_Z_n_p(R, U, V, Omega):
    """ n (|Omega|) and p (|Omega|xK) for all Zij with Mult(Rij,(Ui0Vj0,..,UiKVjK)) prior. """
    K = U.shape[1]
    indices_i, indices_j = zip(*Omega)
    U_list, V_list = U[indices_i,:], V[indices_j,:]
    n_list = R[indices_i, indices_j]
    p_list = U_list * V_list
    p_sum = numpy.repeat(numpy.sum(p_list,axis=1)[:,numpy.newaxis], K, axis=1)
    p_list /= p_sum
    return (n_list, p_list)


''' (Gausian) Gaussian (univariate posterior). '''
def gaussian_gaussian_mu_tau(k, lamb, R, M, U, V, tau):
    """ muUk and tauUk (vectors) for Uk with N(0,I/lamb) prior (I=identity matrix). """
    I, J, K = R.shape[0], R.shape[1], U.shape[1]
    assert R.shape == M.shape and V.shape == (J,K) and U.shape[0] == I
    tauUk = lamb + tau * numpy.sum( M * V[:,k]**2, axis=1)   
    #muUk = 1. / tauUk * ( tau * numpy.sum( 
    #    M * ( ( R - minpy_dot(U,V.T) + minpy_outer(U[:,k],V[:,k])) * V[:,k] ), axis=1) )
    V_ktilde = numpy.append(V[:,:k],V[:,k+1:],axis=1)
    U_ktilde = numpy.append(U[:,:k],U[:,k+1:],axis=1)
    muUk = 1. / tauUk * ( tau * (
        minpy_dot(M*R, V[:,k]) - minpy_dot(M*minpy_dot(U_ktilde, V_ktilde.T), V[:,k])) )
    assert muUk.shape == (I,) and tauUk.shape == (I,)
    return (muUk, tauUk)


''' (Gausian) Gaussian (multivariate posterior). '''
def gaussian_gaussian_mu_sigma(lamb, Ri, Mi, V, tau):
    """ mu and sigma for Ui with N(0,I/lamb) prior (I=identity matrix). """
    assert Ri.shape == Mi.shape and Ri.shape[0] == V.shape[0]
    K = V.shape[1]
    V_masked = (Mi * V.T).T # zero rows when j not in Mi
    precision = lamb * numpy.eye(K) + tau * ( minpy_dot(V_masked.T,V_masked) )
    sigma = numpy.linalg.inv(precision)
    Ri_masked = Mi * Ri # zero entries when j not in Mi
    mu = minpy_dot(sigma, tau * minpy_dot(Ri_masked, V))
    assert mu.shape[0] == V.shape[1] and sigma.shape == (V.shape[1], V.shape[1])
    return (mu, sigma)
    
#def gaussian_gaussian_mu_sigma(lamb, R, M, V, tau):
#    """ mu (IxKxK) and sigma (IxKxK) for all Ui with N(0,I/lamb) prior (I=identity matrix). """
#    assert R.shape == M.shape and R.shape[1] == V.shape[0]
#    I, K = R.shape[0], V.shape[1]   
#    V_expanded = numpy.repeat(V[numpy.newaxis,:,:], I, axis=0)
#    M_expanded = numpy.repeat(M[:,:,numpy.newaxis], K, axis=2)
#    V_masked = (M_expanded * V_expanded)
#    precision = lamb * numpy.repeat(numpy.eye(K)[numpy.newaxis,:,:], I, axis=0) + \
#                tau * numpy.repeat((numpy.sum(V_masked * V_masked, axis=1))[:,:,numpy.newaxis], K, axis=2)
#    sigma = numpy.array([numpy.linalg.inv(prec) for prec in precision])
#    R_masked = M * R # zero entries when j not in Mi
#    mu = numpy.sum(sigma * tau * numpy.repeat((minpy_dot(R_masked, V))[:,:,numpy.newaxis], K, axis=2), axis=2)
#    assert mu.shape == (I,K) and sigma.shape == (I,K,K)
#    return (mu, sigma)


''' (Gausian) Gaussian + Wishart '''
def gaussian_gaussian_wishart_mu_sigma(muU, sigmaU_inv, Ri, Mi, V, tau):
    """ mu and sigma for Ui with N(muU,sigmaU) prior. """
    assert Ri.shape == Mi.shape and Ri.shape[0] == V.shape[0]
    V_masked = (Mi * V.T).T # zero rows when j not in Mi
    precision = sigmaU_inv + tau * ( minpy_dot(V_masked.T,V_masked) )
    sigma = numpy.linalg.inv(precision)
    Ri_masked = Mi * Ri # zero entries when j not in Mi
    mu = minpy_dot(sigma, minpy_dot(sigmaU_inv, muU) + tau * minpy_dot(Ri_masked, V))
    assert mu.shape[0] == V.shape[1] and sigma.shape == (V.shape[1], V.shape[1])
    return (mu, sigma)

def gaussian_wishart_beta0_v0_mu0_W0(beta0, v0, mu0, W0, U):
    """ beta0_s, v0_s, mu0_s, W0_s for muU, sigmaU with NIW(beta0,v0,mu0,W0) prior. """
    I, K = U.shape
    assert mu0.shape == (K,) and W0.shape == (K, K)
    beta0_s = beta0 + I
    v0_s = v0 + I
    U_bar = numpy.sum(U, axis=0) / float(I) # vector giving average per column of U
    S_bar = minpy_dot(U.T, U) / float(I) # matrix giving covariance of columns of U
    mu0_s = ( beta0 * mu0 + I * U_bar) / ( beta0 + I )
    W0_s = W0 + I * S_bar + (beta0*I)/float(beta0+I) * minpy_outer(mu0-U_bar, mu0-U_bar)
    assert mu0_s.shape == (K,) and W0_s.shape == (K, K)
    return (beta0_s, v0_s, mu0_s, W0_s)


''' (Gausian) Gaussian + Automatic Relevance Determination '''
def gaussian_gaussian_ard_mu_sigma(lamb, Ri, Mi, V, tau):
    """ mu and sigma for Ui with N(0,diag(1/lamb)) prior. lamb is a vector. """
    assert Ri.shape == Mi.shape and Ri.shape[0] == V.shape[0] and lamb.shape[0] == V.shape[1]
    V_masked = (Mi * V.T).T # zero rows when j not in Mi
    precision = numpy.diag(lamb) + tau * ( minpy_dot(V_masked.T,V_masked) )
    sigma = numpy.linalg.inv(precision)
    Ri_masked = Mi * Ri # zero entries when j not in Mi
    mu = minpy_dot(sigma, tau * minpy_dot(Ri_masked, V))
    assert mu.shape[0] == V.shape[1] and sigma.shape == (V.shape[1], V.shape[1])
    return (mu, sigma)

def gaussian_ard_alpha_beta(alpha0, beta0, Uk, Vk):
    """ alpha_s and beta_s for lambdak with Gamma(alpha0,beta0) prior. """
    I, J = Uk.shape[0], Vk.shape[0]
    alpha_s = alpha0 + I / 2. + J / 2.
    beta_s = beta0 + numpy.sum(Uk**2) / 2. + numpy.sum(Vk**2) / 2.
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
    cov_U_ktilde = minpy_dot(U_ktilde.T, U_ktilde)
    V_ktilde = numpy.append(V[:,:k],V[:,k+1:],axis=1)
    
    # If K=1, the VP prior bit has no effect
    tauUik = tau*numpy.sum(Mi*V[:,k]**2)
    if K > 1: 
        D_ktilde_ktilde = numpy.linalg.det(cov_U_ktilde)
        A_ktilde_ktilde = adjugate_matrix(cov_U_ktilde)
        assert cov_U_ktilde.shape == (K-1,K-1) and A_ktilde_ktilde.shape == (K-1,K-1)
        tauUik += gamma * (D_ktilde_ktilde - minpy_dot(minpy_dot(U_i_ktilde,A_ktilde_ktilde),U_i_ktilde))
        muUik = 1./tauUik * (
            tau * (minpy_dot(Mi*Ri, V[:,k]) - minpy_dot(minpy_dot(U_i_ktilde, V_ktilde.T), Mi*V[:,k])) +
            gamma * minpy_dot(minpy_dot(U_i_ktilde,A_ktilde_ktilde), minpy_dot(U_itilde_ktilde.T,U_itilde_k))
        )
        #muUik += 1./tauUik * (
        #    tau * numpy.sum(Mi *((Ri-minpy_dot(U[i,:],V.T)+U[i,k]*V[:,k])*V[:,k])) +
        #    gamma * minpy_dot(minpy_dot(U_i_ktilde,A_ktilde_ktilde), minpy_dot(U_itilde_ktilde.T,U_itilde_k)) )
    else:
        muUik = 1./tauUik * ( 
            tau * (minpy_dot(Mi*Ri, V[:,k]) - minpy_dot(minpy_dot(U_i_ktilde, V_ktilde.T), Mi*V[:,k])) ) 
        #muUik = 1./tauUik * (
        #    tau * numpy.sum(Mi *((Ri-minpy_dot(U[i,:],V.T)+U[i,k]*V[:,k])*V[:,k])) )
    return (muUik, tauUik)


''' (Gausian) Exponential '''
def gaussian_exponential_mu_tau(k, lamb, R, M, U, V, tau):
    """ muUik and tauUik for Uik with Exp(lamb) prior. 
        We do updates per column of U (so Uk). """
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    assert U.shape[1] == V.shape[1]
    tauUk = tau * numpy.sum( M * V[:,k]**2, axis=1)
    #muUk = 1. / tauUk * ( -lamb + tau * numpy.sum(M * ( (R-minpy_dot(U,V.T)+minpy_outer(U[:,k],V[:,k]))*V[:,k] ), axis=1))
    V_ktilde = numpy.append(V[:,:k],V[:,k+1:],axis=1)
    U_ktilde = numpy.append(U[:,:k],U[:,k+1:],axis=1)
    muUk = 1. / tauUk * ( -lamb + tau * (
        minpy_dot(M*R, V[:,k]) - minpy_dot(M*minpy_dot(U_ktilde, V_ktilde.T), V[:,k])) )
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
    beta_s = beta0 + numpy.sum(Uk) + numpy.sum(Vk)
    return (alpha_s, beta_s)


''' (Gausian) Truncated Normal '''
def gaussian_tn_mu_tau(k, muU, tauU, R, M, U, V, tau):
    """ mu and tau for Uik with TN(muU,tauU) prior. 
        We do updates per column of U (so Uk). """
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    assert U.shape[1] == V.shape[1]
    tauUk = tauU + tau * numpy.sum( M * V[:,k]**2, axis=1)
    #muUk = 1. / tauUk * ( muU * tauU + tau * numpy.sum(M * ( (R-minpy_dot(U,V.T)+minpy_outer(U[:,k],V[:,k]))*V[:,k] ), axis=1))
    V_ktilde = numpy.append(V[:,:k],V[:,k+1:],axis=1)
    U_ktilde = numpy.append(U[:,:k],U[:,k+1:],axis=1)
    muUk = 1. / tauUk * ( muU * tauU + tau * (
        minpy_dot(M*R, V[:,k]) - minpy_dot(M*minpy_dot(U_ktilde, V_ktilde.T), V[:,k])) )
    assert tauUk.shape == muUk.shape    
    return (muUk, tauUk)


''' (Gausian) Truncated Normal + hierarchical '''
def gaussian_tn_hierarchical_mu_tau(k, muUk, tauUk, R, M, U, V, tau):
    """ mu and tau for Uik with TN(muUik,tauUik) prior, with hierarchical prior 
        for muUik, tauUik. We do updates per column of U (so Uk). """
    return gaussian_tn_mu_tau(k=k, muU=muUk, tauU=tauUk, R=R, M=M, U=U, V=V, tau=tau)

def tn_hierarchical_mu_m_t(mu_mu, tau_mu, U, tauU):
    """ m and t for mu^U_ik with hierarchical prior (hyperparams mu_mu, tau_mu).
        We compute the values for all i,k; so U, tauU should all be a matrix. """
    assert U.shape == tauU.shape
    t = tau_mu + tauU
    m = 1. / t * ( tauU * U + mu_mu * tau_mu )
    assert m.shape == U.shape and t.shape == U.shape
    return (m, t)

def tn_hierarchical_tau_a_b(a, b, U, muU):
    """ a_s and b_s for tau^U_ik with hierarchical prior (hyperparams a, b).
        We compute the values for all i,k; so U, muU should both be a matrix. """
    assert U.shape == muU.shape
    a_s = a * numpy.ones(U.shape) + 0.5
    b_s = b + ( U - muU )**2 / 2.
    return (a_s, b_s)


''' (Gausian) Half Normal '''
def gaussian_hn_mu_tau(k, sigma, R, M, U, V, tau):
    """ mu and tau for Uik with HN(sigma) prior. 
        We do updates per column of U (so Uk). """
    assert R.shape == M.shape and R.shape[0] == U.shape[0] and R.shape[1] == V.shape[0]
    assert U.shape[1] == V.shape[1]
    tauUk = 1. / sigma**2 + tau * numpy.sum( M * V[:,k]**2, axis=1)
    #muUk = 1. / tauUk * ( tau * numpy.sum(M * ( (R-minpy_dot(U,V.T)+minpy_outer(U[:,k],V[:,k]))*V[:,k] ), axis=1))
    V_ktilde = numpy.append(V[:,:k],V[:,k+1:],axis=1)
    U_ktilde = numpy.append(U[:,:k],U[:,k+1:],axis=1)
    muUk = 1. / tauUk * ( tau * (
        minpy_dot(M*R, V[:,k]) - minpy_dot(M*minpy_dot(U_ktilde, V_ktilde.T), V[:,k])) )
    assert tauUk.shape == muUk.shape   
    return (muUk, tauUk)


''' (Poisson) Gamma '''
def poisson_gamma_a_b(a, b, Mi, Vk, Zik):
    """ a_s and b_s for Uik with Gamma(a,b) prior. """
    a_s = a + minpy_dot(Mi, Zik) #numpy.sum(Mi * Zik) #
    b_s = b + minpy_dot(Mi, Vk) #numpy.sum(Mi * Vk) #
    return (a_s, b_s)
    
#def poisson_gamma_a_b(a, b, M, V, Z):
#    """ a_s (IxK) and b_s (IxK) for all Uik with Gamma(a,b) prior. """
#    I, J, K = Z.shape
#    M_repeat_K = numpy.repeat(M[:,:,numpy.newaxis], K, axis=2)
#    V_repeat_I = numpy.repeat(V[numpy.newaxis,:,:], I, axis=0)
#    a_s = a + numpy.sum(M_repeat_K * Z, axis=1)
#    b_s = b + numpy.sum(M_repeat_K * V_repeat_I, axis=1)
#    return (a_s, b_s)
    
    
''' (Poisson) Gamma + hierarchical '''
def poisson_gamma_hierarchical_a_b(a, hUi, Mi, Vk, Zik):
    """ a_s and b_s for Uik with Gamma(a,h^U_i) prior, and h^U_i ~ Gamma(ap,ap/bp). """
    return poisson_gamma_a_b(a=a, b=hUi, Mi=Mi, Vk=Vk, Zik=Zik)

def gamma_hierarchical_hUi_a_b(ap, bp, a, Ui):
    """ a_s and b_s for h^U_i with Gamma(ap,ap/bp) prior, and Uik ~ Gamma(a,h_i^U). """
    K = Ui.shape[0]
    a_s = ap + K * a
    b_s = ap / float(bp) + numpy.sum(Ui)
    return (a_s, b_s)


''' (Poisson) Dirichlet '''
def poisson_dirichlet_alpha(alpha, Mi, Zi):
    """ alpha (vector) for Ui with Dir(alpha) prior in Poisson models. """
    assert Mi.shape[0] == Zi.shape[0] and alpha.shape[0] == Zi.shape[1]
    alpha_s = alpha + numpy.sum(Mi * Zi.T, axis=1)
    assert alpha_s.shape == alpha.shape
    return alpha_s