import numpy as np
import scipy as scp
import scipy.linalg as scpla
import numpy.linalg as npla

def markov_app(data,tt,dim,n2,twidth,r,g,tol):
    """ Computes a Markovian approximation to a given ACF by first computing a Prony series
        by performing a rational approximation and a constrained least-squares optimization, followed by
        a construction of the system matrix and computation of the Covariance matrix and the
        direction of the Brownian motion.

    Parameters:
    data (d x d x N real array): data of the ACF
    tt (N x 1 real array): time grid of the ACF
    dim (int): dimension of the system
    n2 (int): number of sample points
    twidth(int): spacing between the time grid points
    r, g (int): radius and number of grid points of equiangular grid
    tol (int): tolerance for AAA method

    Returns:
    A (m x m real array): system matrix of the system with m-2d auxiliary variables
    Sigma (m x m real array): symmetric and positive definite covariance matrix of the system
    G (m x d real array): direction of the Brownian motion
    lambdaj (m x 1 complex array): exponents of the computed Prony series
    res (d x d x m complex array): coefficients of the computed Prony series
    phi (d x d x N real array): transformed data such that data(t=0) = I

    """

    ## Rescaling the data such that phi(0) = I
    L = scp.linalg.cholesky(data[:,:,0])
    phi = np.zeros_like(data)
    for i in range(1,tt.size):
        phi[:,:,i] = npla.inv(L.T)@data[:,:,i]@npla.inv(L)
    phi[:,:,0] = np.identity(dim)

    ## Sampling the data
    t = tt[0:twidth*n2:twidth]
    y = phi[:,:,0:twidth*n2:twidth]
    tau = t[1]

    ## Computing the input data for the AAA scheme
    omega, F = generatingFunction(y,r,g,dim,n2)
    poles = blockaaa(omega,F,dim,tol)

    lambdaj = calc_exponents(poles,tau)
    m = lambdaj.size

    ## Optimization of Gammaj
    res = solve_constr_lsq(lambdaj,y,t,dim,m)

    ## Markov parameters
    A = constr_sysmat(lambdaj,res,dim,m)
    m = A.shape[0]
    Sigma, G = solve_Lure(A,dim)

    return A, Sigma, G, lambdaj, res, phi

def blockaaa(Z,F,d,tol):
    """ Computes a rational approximation to a given data set using a matrix-valued version of the
        AAA algorithm. After determining a good approximation, the poles of this rational approximation are computed

    Parameters:
    Z (g x 1 complex array): grid points of the equiangular grid
    F (d x d x g complex array): data points
    d (int): dimension of the system
    tol (int): termination criterion for the iterative scheme

    Returns:
    poles (n x 1): poles of the rational approximation (all poles lie inside the unit circle)

    """
    M = Z.size
    J = np.arange(M)
    z = []
    f = np.zeros((2,2,2),complex)

    # select real data for initial approximation
    z.append(Z[0]), z.append(Z[M//2])
    f[:,:,0] = F[:,:,0]
    f[:,:,1] = F[:,:,M//2]
    J = np.delete(J,[0,M//2])
    m = 2

    L = constr_Loewner(J,F,f,Z,z,d,m)
    Vh = npla.svd(L, full_matrices=False, compute_uv=True)[-1]
    V = np.conjugate(Vh.T)
    w1 = V[:,-1]
    w = symm_weights(w1)

    R = compute_app(w,F,f,Z,z,J)
    errvec = npla.norm(F-R,'fro',axis=(0,1))
    ind = np.argmax(errvec)
    err = np.inf

    while( m <= M//2 and err > tol ):
        # Choose next support point in greedy fashion + complex conjugate point
        m += 2
        cc = M - ind
        z.append(Z[ind]), z.append(Z[cc])
        f = np.dstack((f,F[:,:,ind]))
        f = np.dstack((f,F[:,:,cc]))
        J = np.delete(J,[np.where(J==ind),np.where(J==cc)])
        # Solve optimization problem for the weights
        L = constr_Loewner(J,F,f,Z,z,d,m)
        Vh = npla.svd(L, full_matrices=False, compute_uv=True)[-1]
        V = np.conjugate(Vh.T)
        w1 = V[:,-1]
        w = symm_weights(w1)

        R = compute_app(w,F,f,Z,z,J)
        errvec = npla.norm(F-R,'fro',axis=(0,1))
        ind = np.argmax(errvec)
        err = np.max(errvec)

    return calc_poles(z,w,m)

def generatingFunction(y,r,g,d,N):
    """ Evaluates the generating function on an equiangular grid for a given set of data points.
    
    Parameters:
    y (d x d x N real array): data points
    d, N (int): dimension of the system and number of data points
    r, g (int): radius and number of grid points for equiangular grid

    Returns:
    omega (g x 1 complex array): equiangular grid
    F (d x d x g complex array): data points of the generating function

    """

    omega = r*np.exp(2*np.pi*1j*np.arange(g)/g)
    F = np.zeros((d,d,g),complex)
    for k in range(0,g):
        omegatmp = omega[k]**(-np.arange(1,N+1))
        F[:,:,k] = np.inner(y,omegatmp)
    return omega, F

def constr_Loewner(J,F,f,Z,z,d,m):
    """ Constructs a Loewner matrix for a given set of interpolation and data points.

    Parameters:
    J (g-m x 1 int array): index vector of non-interpolated data points
    F (d x d x g complex array): data points
    f (d x d x m complex array: interpolated data points
    Z (g x 1 complex array): equiangular grid
    z (m x 1 complex array): interpolation points
    d, m (int): dimension of the system and number of interpolation points

    Returns:
    L (d*d*(g-m) x m complex array): Loewner matrix 
    
    """
    L = np.zeros((J.size*d**2,m),complex)
    for k in range(0,m):
        tmp = np.zeros((J.size*d,d),complex)
        for l in range(0,J.size):
            tmp[l*d:(l+1)*d,:] = ( F[:,:,J[l]] - f[:,:,k] ) / ( Z[J[l]] - z[k] )
        L[:,k] = tmp.flatten(order='F')
    return L 

def symm_weights(w1):
    """ Symmetrizes a computed set of weights, such that the resulting weight vector leads
        to a real-valued rational approximation.

    Parameters:
    w1 (m x 1 complex array): weight vector with non complex conjugate entries

    Returns:
    w (m x 1 complex array): weight vector with complex conjugate entries
    
    """
    n = w1.size
    w2 = np.copy(w1)
    for i in range(2,n,2):
        w2[i] = w1[i+1]
        w2[i+1] = w1[i]

    if npla.norm(w1 + w2) > 1e-12:
        w = w1 + np.conjugate(w2)
        w = w/npla.norm(w)
    else:
        w = 1j*w1/npla.norm(w1)
    
    return w

def compute_app(w,F,f,Z,z,J):
    """ Evaluates a rational function in barycentric form for a given set of grid points.

    Parameters:
    z, w (m x 1 complex array): interpolation points and weights of rational approximation
    f (d x d x m complex array): function values at z
    F (d x d x g complex array): data points at Z
    Z (g x 1 complex array): grid points of the equiangular grid
    J (g-m x 1 int array): index vecto

    Returns:
    R (d x d x g): rational approximation evaluated at the given grid points
    
    """
    R = np.copy(F)
    for k in range(0,J.size):
        cauchyw = np.divide(w,Z[J[k]]-z)
        N = np.inner(f,cauchyw)
        D = sum(cauchyw)
        R[:,:,J[k]] = N / D
    return R

def calc_poles(z,w,m):
    """ Computes the poles of a rational approximation in barycentric form by solving a (m+1) x (m+1) 
        generalized eigenvalue problem. Two eigenvalues are per se infinite, the remaining (m-1) eigenvalues
        are the poles of the rational approximation. Poles outside the unit disk are discarded.
        

    Parameters:
    z, w (m x 1 complex array): interpolation points and weights of the rational approximation
    m (int): number of interpolation points

    Returns:
    poles (n x 1 complex array): poles of the rational approximation. 

    """

    B = np.identity(m+1)
    B[0,0] = 0
    E = np.zeros((m+1,m+1),complex)
    E[0,1:] = w
    E[1:,0] = np.ones((m))
    E[1:,1:] = np.diagflat(z)
    poles = scpla.eigvals(E,B)

    return poles[np.where(np.abs(poles)<1)]

def calc_exponents(pol,tau):
    """ Computes the exponents from a given set of poles and time width.
        Negative real poles are duplicated to complex conjugate pairs.

    Parameters:
    pol (n x 1 complex array): contains the poles of a rational function
    tau (int): time width used for sampling of the ACF

    Returns:
    lambdaj (m x 1 complex array): computed exponents. In general we have n = m,
                                   in case of negative real poles, we have m > n.
    
    """
    pospol = pol[np.logical_and(np.real(pol)>0,np.abs(np.imag(pol))<1e-8)]
    negpol = pol[np.logical_and(np.real(pol)<0,np.abs(np.imag(pol))<1e-8)]
    ccpol = np.delete(pol,np.logical_or(np.logical_and(np.real(pol)>0,np.abs(np.imag(pol))<1e-8),
                                        np.logical_and(np.real(pol)<0,np.abs(np.imag(pol))<1e-8)))
    
    pol = np.concatenate((np.log(pospol),np.sort(np.log(ccpol))))
    for x in negpol:
        pol = np.append(pol,np.log(x)+1j*np.pi)
        pol = np.append(pol,np.log(x)-1j*np.pi)
    """
    pol = np.sort_complex(pol)
    negpol = pol[np.logical_and(np.real(pol)<0,np.abs(np.imag(pol))<1e-8)]
    pol = np.delete(pol,np.where(np.logical_and(np.real(pol)<0,np.abs(np.imag(pol))<1e-12)))
    for x in negpol:
        pol = np.append(pol,x+1j*np.pi)
        pol = np.append(pol,x-1j*np.pi)"""
    return pol/tau

def solve_constr_lsq(lambdaj,y,t,dim,m):
    """ Solves the optimization problem for the coefficients Gammaj

    Parameters:
    lambdaj (m x 1 complex array): exponents of the Prony series
    y (d x d x N real array): data of the underlying ACF
    t (N x 1 real array): time grid of the data
    dim, m (int): dimension of the underlying system and number of exponents

    Returns:
    res (d x d x m complex array): coefficients Gammaj of the Prony series
    
    """

    # Set up permutation matrix
    I = np.identity(dim**2)
    indmat = np.arange(0,dim**2).reshape((dim,dim),order='F')
    permut = indmat.T
    P = I[permut.flatten(order='F')]
    upperT = np.triu(indmat,1)
    evenIndex = upperT[np.where(upperT!=0)]

    y0 = y[:,:,0].flatten(order='F')

    # Constraint matrices
    M0 = np.kron(np.ones((1,m)),I)
    M1 = np.kron(lambdaj.T,I)
    M2 = 1j*np.kron(lambdaj.T**2,I-P)
    M3 = np.kron(lambdaj.T**3,I+P)

    # Constraints
    B = np.concatenate((M0,M1,M2[evenIndex,:]),axis=0)
    d = np.concatenate((y0,np.zeros(dim**2+evenIndex.size)),axis=0)

    # Solve constrained LSQ
    G = np.kron(np.exp(np.outer(t,lambdaj)),I)
    a = constr_lsq(G,y.flatten(order='F'),B,d)

    # Check if semidefinite constraint is satisfied
    D = npla.eigvals(np.reshape(M3@a,(dim,dim),order='F'))
    D = np.delete(D,np.where(np.abs(D)<1e-10))

    if np.min(np.real(D)) < 0:
        Warning('Semidefiniteness constraint not satisfied')

    return a.reshape((dim,dim,m),order='F')

def constr_lsq(A,b,B,d):
    """ Solves a constrained LSQ problem of the form
            min ||Ax-b||_2 subject to Bx = d
        using the nullspace method presented in the book by Bjoerck
    
    Parameters:
    A (q x n complex array): matrix of LSQ
    B (p x n complex array): constraint matrix with n >= p
    b (q x 1 real array): rhs of LSQ
    d (p x 1 real array): rhs of contraints

    Returns:
    x (n x 1 complex array): solution of constrained LSQ problem

    """

    # Check if lsq problem has unique solution
    S = npla.svd(np.concatenate((A,B),axis=0).T)[1]
    if S[-1] < 1e-14:
        Warning("Constrained LSQ has no unique minimizer")

    p, n = B.shape
    QB, RB = npla.qr(np.conjugate(B.T),mode="complete")
    Q1 = QB[:,0:p]
    Q2 = QB[:,p:]
    x1 = Q1@npla.solve(np.conjugate(RB[0:p,0:p].T),d)
    QA, RA = npla.qr(A@Q2,mode="complete")
    c = np.conjugate(QA.T)@(b-A@x1)
    y2 = npla.solve(RA[0:n-p,0:n-p],c[0:n-p])

    return x1 + Q2@y2

def constr_sysmat(exps,res,d,m):
    """ Constructs a system matrix from given exponents lambdaj and coefficients Gammaj.
        The Jordan canonical form J is set up and transformed to system matrix A via a
        similarity transformation X

    Parameters:
    exps (N x 1 complex array): exponents lambdaj of the Prony series
    res (d x d x N complex array): coefficients Gammaj of the Prony series
    d,m (int): dimension of the system and the number of exponents
    
    Returns:
    A (dm x dm real array): system matrix of a d-dimensional Markovian system. The size of the matrix
                            is dm x dm in general but might be smaller due to rank deficient coefficients

    """

    r = 0
    counter = 0
    J = np.zeros((d*m,d*m))
    USigma = np.zeros((d,d*m))
    VSigma = np.zeros_like(USigma.T)

    # real exponents first
    while np.abs(np.imag(exps[r])) < 1e-8:
        U,S,V = npla.svd(res[:,:,r])
        for i in range(0,d):
            if S[i] < 1e-12:
                break
            J[counter,counter] = np.real(exps[r])
            USigma[:,counter] = np.sqrt(S[i])*np.real(U[:,i])
            VSigma[counter,:] = np.sqrt(S[i])*np.real(np.conjugate(V)[i,:])
            counter += 1
        r += 1
        if r > m:
            break

    # complex conjugate pairs next
    while r < m:
        U,S,V = npla.svd(res[:,:,r])
        for i in range(0,d):
            if S[i] < 1e-12:
                break
            J[counter:counter+2,counter:counter+2] = [[np.real(exps[r]),np.imag(exps[r])],[-np.imag(exps[r]),np.real(exps[r])]]
            USigma[:,counter] = np.sqrt(2*S[i])*np.real(U[:,i])
            USigma[:,counter+1] = np.sqrt(2*S[i])*np.imag(U[:,i])
            VSigma[counter,:] = np.sqrt(2*S[i])*np.real(V[i,:])
            VSigma[counter+1,:] = -np.sqrt(2*S[i])*np.imag(V[i,:])
            counter += 2
        r += 2

    X = basis_trafo(USigma.T,VSigma)
    return X@J[0:counter,0:counter]@npla.inv(X)

def basis_trafo(U,V):
    """ Computes a similarity transformation such that X*V = E_dS^(1/2) and X.T*E_d = US^(-1/2)
        with S = U*V
    
    Parameters:
    U,V (N x p real array): matrices U and V for basis transformation, see above

    Returns:
    X (N x N real array): basis transformation matrix that satisfies identity from above

    """

    d = V.shape[1]
    S = U.T@V
    Q = npla.svd(V)[0]
    X = np.concatenate((npla.inv(scpla.sqrtm(S))@U.T,np.conjugate(Q[:,d:].T)),axis=0)
    return X

def solve_Lure(A,d):
    """ Solves the singular Lur'e equations
            A*Sigma + Sigma*A.T = -G*G.T
        for Sigma and G. The system is reduced to a regular system in the first step.
        The regular system is solved via solving a CARE and the solution is transformed back
        to the full system.

    Parameters:
    A (N x N real array): system matrix A, which is partitioned into 4 matrix blocks, respectively
    d (int): dimension of the underlying system

    Returns:
    Sigma (N x N real array): covariance matrix of the system (Solution of the Lyapunov equation above)
    G (N x d real array): direction of the Brownian motion (rhs of the Lyapunov equation above)
    
    """

    # partition A
    B = A[0:d,d:].T
    C = -A[d:,0:d]
    A0 = A[d:,d:]

    N = A0.shape[0]

    # perform reduction step
    X = basis_trafo(B,C)
    #XAX = X@npla.solve(A0,X)
    XAX = X@A0@npla.inv(X)
    D1 = -XAX[0:d,0:d]
    B1 = XAX[0:d,d:].T
    C1 = -XAX[d:,0:d]
    A1 = XAX[d:,d:]

    R1 = D1 + D1.T

    # check if reduced system is solvable
    if np.min(npla.eigvals(R1)) < 0:
        print('Approximation is not positive real!\n D of reduced system is not positive definite.')
        return np.zeros((N+d,N+d)), np.zeros((N+d,d))

    if A1.shape[0] > 0 and np.max(np.real(npla.eigvals(A1))) >= 0:
        print('Approximation is not positive real!\n A0 of reduced system is not stable.')
        return np.zeros((N+d,N+d)), np.zeros((N+d,d))
    
    # Solve reduced regular Lur'e system
    P = A1 - C1@npla.inv(R1)@B1.T
    Q = C1@npla.inv(R1)@C1.T

    Sigma1 = scpla.solve_continuous_are(P.T,B1,Q,-R1,e=None,s=None)
    # check error
    if np.allclose(P@Sigma1 + Sigma1@P.T + Sigma1@B1@npla.inv(R1)@B1.T@Sigma1,-Q,rtol=1e-6) == False:
        print('Riccati equation of reduced system not satisfied!')
        return np.zeros((N+d,N+d)), np.zeros((N+d,d))
    
    # check symmetry
    if Sigma1.shape[0] > 0 and np.allclose(Sigma1,Sigma1.T,atol=1e-5)==False:
        print('Covariance matrix Sigma0 not symmetric!')
        return np.zeros((N+d,N+d)), np.zeros((N+d,d))
    
    # transform reduced system back to full system
    K1 = npla.cholesky(R1)
    L1 = (C1-Sigma1@B1)@npla.inv(K1.T)

    XL = np.concatenate((K1,L1),axis=0)

    L = npla.solve(X,XL)
    XSigma0X = np.identity(N)
    XSigma0X[d:,d:] = Sigma1
    Sigma0 = npla.inv(X)@XSigma0X@npla.inv(X.T)
    Sigma0 = (Sigma0 + Sigma0.T)/2 # nullify rounding errors

    Sigma = np.identity(N+d)
    Sigma[d:,d:] = Sigma0

    G = np.concatenate((np.zeros((d,d)),L),axis=0)

    return Sigma, G



