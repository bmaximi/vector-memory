#= This file implements functions for solving both the regular and the singular 
Lur'e equations. =#

include("riccati_aux.jl")

struct RiccatiError <: Exception
    msg::String
end

function riccati(F::AbstractMatrix{T}, G::AbstractMatrix{T}, H::AbstractMatrix{T}; 
        tol_sol::Real=1e-6, tol_imag::Real=1e-10, tol_symm::Real=1e-10, 
        warn::Bool=warn) where T <: Real
    #= Algorithm from Alan J. Laub, 
        A SCHUR METHOD FOR SOLVING ALGEBRAIC RICCATI EQUATIONS, 1978, 
    for solving the Riccati equation
      F^TX + XF - XGX + H = 0
    Notation of variables is inspired by this article.
    Important: The conditions from this article are not satisfied here since G is negative 
    semidefinite. Hence the equation does not necessarily have a solution.
    To test if a solution eixsts, one has to check if pure imaginary eigenvalues exist,
    using the algorithm from
    Peter, Benner, Volker Mehrmann, Hongguo Xu
        A numerically stable, structure preserving method for computing the eigenvalues of 
        real Hamiltonian or symplectic pencils, 1998.
    (Note that the periodic QR algorithm mentioned is not necessary and can be replaced by
    any eigenvalue algorithm which guarantees that the computed eigenvalues of a real matrix
    are symmetric w.r.t. the real axis, hence we can use the standard function eigvals below.)
    =#
    n = size(F, 1)
    Z = [F -G; -H -F']
    Z_hessenberg = special_hessenberg_transformation(Z)[1]
    Z_vals_temp = sqrt.(-eigvals(Z_hessenberg[n+1:end, n+1:end]'*Z_hessenberg[1:n, 1:n]))
    Z_vals = [Z_vals_temp; -Z_vals_temp]
    if !all(sort(Z_vals, by=real) + sort(Z_vals, by=real, rev=true) .== 0)
        println("Eigenvalues of Z are not exactly symmetric w.r.t. to imaginary axis!")
        println("Eigenvalues: $(sort(Z_vals, by=real))")
    end
    Z_schur = schur(Z)
    pos_eig = real.(Z_schur.values) .<= 0
    if warn
        #= The employed eigenvalue algorithm asserts that the obtained eigenvalues are exactly
        symmetric w.r.t. the imaginary axis; hence testing for exact equality is possible. =#
        if any(real.(Z_vals) .== 0)
            println("Z has $(sum(real.(Z_vals) .== 0)) pure imaginary eigenvalues. "*
                "The Riccati equation is probably not solvable.")
        elseif minimum(abs.(real.(Z_vals))) < tol_imag
            #= Output for testing purposes: If Z has a pure imaginary eigenvalue, the Ricatti
            equation usually has no solution. =#
            println("Z has nearly imaginary eigenvalues. Smallest absolute real part of an "*
                "eigenvalue: ", fc(minimum(abs.(real.(Z_vals)))))
                println("The Riccati equation is probably not solvable.")
        end
    end
    if n != sum(pos_eig)
        if warn
            println("Eigenvalues of Z: $(fc(eigvals(Z)))")
        end
        # This should happen only if one of the above error messages was already issued.
        throw(RiccatiError("Riccati algorithm failed"))
    end
    ordschur!(Z_schur, pos_eig)
    U = Z_schur.vectors
    X = U[n+1:end, 1:n] / U[1:n, 1:n]
    #= Only the absolute error is considered here without taking into regard the orders
    of magnitude of the matrices' entries; come kind of relative error would be better. =#
    if warn && !isempty(Z) && maximum(abs.(F'*X + X*F - X*G*X + H)) > tol_sol
        println("Large numerical error in Riccati solution: ",
            fc(maximum(abs.(F'*X + X*F - X*G*X + H))))
    end
    if warn && size(X, 1) > 0 && maximag(X) / maximum(abs.(real.(X))) > tol_imag
        println("Riccati matrix X is not real. Maximum imaginary part: ", fc(maximag.(X)))
    end
    if warn
        if size(X, 1) > 0 && maximum(abs.(X-X')) > tol_symm
            println("Riccati solution X is not (near-)symmetric: "*
                "maximum(abs.(X-X')) = $(fc(maximum(abs.(X-X'))))")
        elseif minimum(real(eigvals(X+X'))) < 0
            println("Riccati solution X is not positive semi-definite.")
        end
    end
    return real(X)
end

function riccati_matrices(A::AbstractMatrix{<:Real}, d::Integer, vacf_0=I)
    A0 = A[d+1:end, d+1:end]
    B = A[1:d, d+1:end]'
    C = -A[d+1:end, 1:d]
    D = -A[1:d, 1:d]
    return riccati_matrices(A0, B, C, D, vacf_0)
end
function riccati_matrices(A0::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real}, 
        C::AbstractMatrix{<:Real}, D::AbstractMatrix{<:Real}, vacf_0=I)
    C = C * vacf_0
    D = D * vacf_0
    D_lu = lu(D+D')
    #= If D is small, multiplication with inv(D) can signifcantly increase the error =#
    F = A0'-B*(D_lu\C')
    G = -B*(D_lu\B')
    H = C*(D_lu\C')
    return F, 0.5*(G+G'), 0.5*(H+H')
end


function lure_regular(A::AbstractMatrix{<:Real}, d::Integer, vacf_0=I; kw...)
    A0 = A[d+1:end, d+1:end]
    B = A[1:d, d+1:end]'
    C = -A[d+1:end, 1:d]
    D = -A[1:d, 1:d]
    return lure_regular(A0, B, C, D, vacf_0; kw...)
end
# The function lure_regular solves the regular Lur'e equations via the function riccati.
function lure_regular(A0::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real}, 
        C::AbstractMatrix{<:Real}, D::AbstractMatrix{<:Real}, vacf_0=I; 
        tol_sol::Real=1e-6, warn::Bool=warn)
    C = C * vacf_0
    D = D * vacf_0
    try
        if size(A0, 1) > 0 && !isstable(A0)
            throw(RiccatiError("A_0 is not stable"))
        end
        if !isstable(-D-D')
            throw(RiccatiError("-D-D' is not negative definite (eigenvalues: $(fc(eigvals(D+D'))))"))
        end
        X = riccati(riccati_matrices(A0, B, C, D)..., warn=warn)
    catch e
        if isa(e, RiccatiError)
            if warn
                println("Riccati equation has no solution:")
                println(e.msg)
            end
        end
        rethrow()
    end
    #= X should actually be symmetric, therefore the following line should only correct 
    numerical errors. =#
    X = 0.5*(X+X')
    W = cholesky(D+D').L
    L = (C-X*B)/W'
    G = [W; L]
    if size(X, 1) > 0
        err = maximum(abs.(A0*X+X*A0'+L*L'))
        if warn && err > tol_sol
            println("Large numerical error in Lur'e solution: ", fc(err))
        end
    end
    #= X and G = [W; L] solve the Lur'e equations:
        X*A' + A*X = -L*L'
        X*B-C = -L*W'
        D + D' = W * W'
    with W = cholesky(D+D').L and L = (C-X*B)/cholesky(D+D') =#
    return X, G
end


function lure(A::AbstractMatrix{<:Real}, d::Integer, vacf_0=I; kw...)
    #= wrapper function for lure_singular_strict which rearranges the matrix entries in 
    a way as they are required by lure_singular_strict 
    (cf. Wang, Speyer, Weiss, System Characterization of Positive Real Conditions, 1990) =#
    return lure_strict(A[d+1:end, d+1:end]', A[1:d, d+1:end]', 
        A[d+1:end, 1:d]', A[1:d, 1:d]', vacf_0; kw...)
end

#= The function lure_singular_strict solves the regular (det(D) != 0) or singular 
(det(D) == 0) Lur'e equations stricly according to 
Wang, Speyer, Weiss, System Characterization of Positive Real Conditions, 1990
In the regular case, the function lure_regular is called. =#
function lure_strict(A_::AbstractMatrix{T}, B_::AbstractMatrix{T}, 
        C_::AbstractMatrix{T}, D_::AbstractMatrix{T}, vacf_0=I; 
        tol_R::Real=1e-13, warn::Bool=warn) where T <: Real
    n, m = size(B_)
    @assert size(A_) == (n, n) (size(A_), (n, n))
    @assert size(C_) == (m, n) (size(C_), (m, n))
    @assert size(D_) == (m, m) (size(D_), (m, m))
    C_ = vacf_0 * C_
    D_ = D_ * vacf_0

    R_ = D_ + D_'

    R_eigen = eigen(R_)
    Gamma1 = R_eigen.vectors[:, abs.(R_eigen.values) .> tol_R] # m×r
    Gamma2 = R_eigen.vectors[:, abs.(R_eigen.values) .<= tol_R] # m×k
    Gamma = [Gamma1 Gamma2] # m×m
    if isempty(Gamma2' * C_)
        #= Considering this case separately is only necessary because 
        GenericSchur (when using BigFloat) leads to an error. =#
        Q2 = I(n) # n×k
        Q1 = zeros(T, n, 0) # n×(n-k)
    else
        temp = svd(Gamma2' * C_, full=true)
        temp_inv = inv(temp.Vt)
        Q2 = temp_inv[:, [findall(abs.(temp.S) .<= tol_R); size(temp, 1)+1:end]] # n×k
        Q1 = temp_inv[:, findall(abs.(temp.S) .> tol_R)] # n×(n-k)
    end
    k = size(Q1, 2)
    r = m-k
    Q = [Q1 Q2] # n×n
    if k == 0
        Sigma, G = lure_regular([D_ C_; B_ A_]', m, I, warn=warn)
        return Sigma, G # n×n, n×m
    end

    #= Change of basis in order to transform D into a matrix of the form [D_ 0; 0 0].
    Gamma is an orthogonal matrix. =#
    D = Gamma'*R_*Gamma # m×m
    A = Q \ A_ * Q # n×n
    B = Q \ B_ * Gamma # n×m
    C = Gamma' * C_ * Q # m×n

    A11 = A[1:k, 1:k] # k×k
    A12 = A[1:k, k+1:end] # k×(n-k)
    A21 = A[k+1:end, 1:k] # (n-k)×k
    A22 = A[k+1:end, k+1:end] # (n-k)×(n-k)
    B_r = B[:, 1:r] # n×r
    B_s = B[:, r+1:end] # n×k
    B_s1 = B_s[1:k, :] # k×k
    B_s2 = B_s[k+1:end, :] # (n-k)×k
    C_r = C[1:r, :] # r×n
    C_s = C[r+1:end, :] # k×n
    C_s1 = C_s[:, 1:k] # k×k
    A_1 = A22 - (B_s2 / B_s1) * A12 # (n-k)×(n-k)
    # The sign of C here differs from the text.
    B_1 = [A21*B_s1+A22*B_s2-(B_s2/B_s1)*(A11*B_s1+A12*B_s2)  B_r[k+1:end, :]] # (n-k)×m
    C_1 = [-C_s1*A12;  C_r[:, k+1:end]] # m×(n-k)
    R_1 = [-(C_s*A*B_s+B_s'*A'*C_s')  -C_s*B_r+B_s'*C_r';
           -B_r'*C_s'+C_r*B_s  D[1:r, 1:r]] # m×m

    Sigma_1, V_1 = lure_singular_strict(A_1, B_1, C_1, 0.5*R_1, warn=warn) # (n-k)×(n-k), (n+r)×m
    V_1 = [V_1[m+1:end, :]; V_1[1:m, :]]

    Sigma = [-B_s1' \ (C_s1 - B_s2'*Sigma_1*B_s2 / B_s1)  -B_s1' \ B_s2'*Sigma_1;
             -Sigma_1*B_s2 / B_s1  Sigma_1] # n×n

    #= According to the cource, T is given by T = [0 B_s1 0; I B_s2 0; 0 0 I]. 
    This implies V = T' \ V_1 where
    inv(T) = [-B_s2 / B_s1 I 0; inv(B_s1) 0 0; 0 0 I]. =#
    V = [B_s1' \ (-B_s2' * V_1[1:n-k, :] + V_1[n-k+1:n, :]); V_1[1:n-k, :]; V_1[n+1:n+r, :]; zeros(k, m)]

    Sigma = Q' \ Sigma / Q
    G = [Gamma*V[n+1:end, :]; Q*V[1:n, :]]
    return Sigma, G
end

