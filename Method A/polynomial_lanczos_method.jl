#= This file implements the Lanczos algorithm for orthogonal polynomials, which represents 
the core of this implementation of Prony's method. =#

using LinearAlgebra

struct PolynomialLanczosError <: Exception
    msg::AbstractString
    p_size::Int
    q_size::Int
end

poly_bilin_form(M::Array{T, 3}) where T <: Real = 
    poly_bilin_form(collect(Matrix.(eachslice(M, dims=3))))

function poly_bilin_form(M::AbstractVector{<:AbstractMatrix{T}}) where T <: Real
    #= returns the bilinear form used in the Lanczos method as a function 
    All matrices in M have to be quadratic and of equal size. =#
    n = length(M)
    d = size(M[1], 1)
    function test_Phi(p::AbstractVector{T}, q::AbstractVector{T})
        # tests if Phi can be computed for this input
        return cld(length(p), d)+cld(length(q), d)-1 <= n
    end
    function test_Phi(p::AbstractMatrix{T}, q::AbstractMatrix{T})
        # tests if Phi can be computed for this input
        return cld(size(p, 1), d)+cld(size(q, 1), d)-1 <= n
    end
    function Phi(p::AbstractVector{T}, q::AbstractVector{T})
        if cld(length(p), d)+cld(length(q), d)-1 > n
            throw(PolynomialLanczosError("polynomial degrees are too large.", 
                length(p), length(q)))
        end
        res = zero(T)
        i_mod_d = 0
        for i in eachindex(p)
            j_mod_d = 0
            temp = fld(i-1, d)+1
            M_temp = M[temp]
            i_mod_d = i_mod_d == d ? 1 : i_mod_d+1
            for j in eachindex(q)
                if j_mod_d == d
                    j_mod_d = 1
                    temp += 1
                    M_temp = M[temp]
                else
                    j_mod_d += 1
                end
                #= more readable (and equivalent), but slower formulation of the code
                i_mod_d = mod1(i, d)
                j_mod_d = mod1(j, d)
                temp = fld(i-1, d)+fld(j-1, d)+1
                M_temp = M[temp] =#
                res += p[i]*M_temp[i_mod_d, j_mod_d]*q[j]
            end
        end
        return res
    end
    function Phi(p::AbstractMatrix{T}, q::AbstractMatrix{T})
        return [Phi(@view(p[:, i]), @view(q[:, j])) for i=axes(p, 2), j=axes(q, 2)]
    end
    return Phi, test_Phi
end


polynomial_lanczos_method(M::AbstractArray{T, 3}; kw...) where T <: Real = 
    polynomial_lanczos_method(collect(Matrix.(eachslice(M, dims=3))); kw...)

function polynomial_lanczos_method(M::AbstractVector{<:AbstractMatrix{T}}; warn::Bool=true, 
        orthogonality_tol::Union{Nothing, AbstractFloat}=1e-12,
        dtol = T <: Rational ? 0 : sqrt(eps(T))) where T <: Real
    #= algorithm and notation according to Freund, 
    Computation of matrix-valued formally orthogonal polynomials and applications (2000) =#
    m, p = size(M[1])
    bilin, test_bilin = poly_bilin_form(M)

    function gamma(n::Integer)::Int
        for j in Iterators.countfrom(0)
            if n_[j] > n
                return j-1
            end
        end
    end
    function findkey(dict::Dict{K, V}, value::V)::K where {K, V}
        for i in dict
            if i.second == value
                return i.first
            end
        end
    end

    dtol_r = dtol*maximum(A -> opnorm(A, 1), M)
    dtol_l = dtol*maximum(A -> opnorm(A, Inf), M)
    
    #= The entries of phi_h, psi_h, phi, and psi are the polynomial coefficients 
    (starting at the scalar coefficient), where phi_h, psi_h, phi, and psi are regarded 
    as scalar polynomials. =#
    phi_h = Dict(i=>[zeros(T, i); one(T)] for i=0:m-1)
    psi_h = Dict(i=>[zeros(T, i); one(T)] for i=0:p-1)
    phi = Dict{Int, Vector{T}}()
    psi = Dict{Int, Vector{T}}()
    # phi_defl and psi_defl store all polynomials which are discarded due to deflation.
    phi_defl = AutosizeMatrix{T}()
    psi_defl = AutosizeMatrix{T}()
    mu_h = Dict(i=>i for i=0:m-1)
    nu_h = Dict(i=>i for i=0:p-1)
    mu = Dict{Int, Int}()
    nu = Dict{Int, Int}()
    m_c, p_c = m, p
    D_phi = Int[]
    D_psi = Int[]
    ell_phi = ell_psi = 0
    ell = 0
    n_ = Dict(0=>0)

    #= Each entry in Phi or Psi represent one block of polynomials. The entries are matrices:
    The first dimension counts the power of a coefficient (in increasing order),
    the second one lists the single polynomials belonging to this block. 
    (Several polynomials in a block appear when look-ahead is necessary.) =#
    Phi = Dict(0=>zeros(T, 0, 0))
    Psi = Dict(0=>zeros(T, 0, 0))

    Delta = Dict{Int, Matrix{T}}()

    #= compared to the source, the first dimension of U and V is shifted by 1
    and the second by m+1 and p+1 respectively, to achieve 1-based indexing =#
    U = AutosizeMatrix{T}()
    V = AutosizeMatrix{T}()

    try
        for n in Iterators.countfrom(0)
            # 1. deflation of phi
            while true
                # check if deflation is necessary
                deflate = true
                for k in Iterators.countfrom(0)
                    # decide whether to deflate phi
                    if k+1 + cld(length(phi_h[n]), m) > length(M) + 1
                        break # polynomial order is too large
                    end
                    if norm(bilin([zeros(T, k*m, m); I], reshape(phi_h[n], :, 1))) > dtol_r
                        deflate = false
                        break
                    end
                end
                if !deflate
                    break
                end
                if warn
                    #= Note that deflation often happens directly the algorithm terminates. 
                    This is part of the algorithm and no real deflation. To avoid confusion, 
                    this following print statement is commented out. =#
                    # println("Deflation in phi at step $n (starting from 0)")
                end

                phi_defl[1:length(phi_h[n]), m-m_c+1] = phi_h[n]
                if m_c == 1
                    if warn
                        # println("Stopping due to deflation in phi")
                    end
                    warn_orthogonality(bilin, Psi, Phi, orthogonality_tol)
                    return Phi, Psi, U, V, Delta, bilin, n_, mu, nu, D_phi, D_psi, phi_defl, psi_defl
                end
                if n >= m_c
                    push!(D_phi, gamma(n-m_c))
                end
                for i = n:n+m_c-2
                    phi_h[i] = phi_h[i+1]
                    mu_h[i] = mu_h[i+1]
                end
                delete!(phi_h, n+m_c-1)
                delete!(mu_h, n+m_c-1)
                m_c -= 1
            end

            # 2. deflation of psi
            while true
                # check if deflation is necessary
                deflate = true
                for k in Iterators.countfrom(0)
                    # decide whether to deflate psi
                    if cld(length(psi_h[n]), p) + k+1 > length(M) + 1
                        break # polynomial order is too large
                    end
                    if norm(bilin(reshape(psi_h[n], :, 1), [zeros(T, k*p, p); I])) > dtol_l
                        deflate = false
                        break
                    end
                end
                if !deflate
                    break
                end
                if warn
                    #= Note that deflation often happens directly the algorithm terminates. 
                    This is part of the algorithm and no real deflation. To avoid confusion, 
                    this following print statement is commented out. =#
                    # println("Deflation in psi at step $n (starting from 0)")
                end

                psi_defl[1:length(psi_h[n]), p-p_c+1] = psi_h[n]
                if p_c == 1
                    if warn
                        # println("Stopping due to deflation in psi")
                    end
                    warn_orthogonality(bilin, Psi, Phi, orthogonality_tol)
                    return Phi, Psi, U, V, Delta, bilin, n_, mu, nu, D_phi, D_psi, phi_defl, psi_defl
                end
                if n >= p_c
                    push!(D_psi, gamma(n-p_c))
                end
                for i = n:n+p_c-2
                    psi_h[i] = psi_h[i+1]
                    nu_h[i] = nu_h[i+1]
                end
                delete!(psi_h, n+p_c-1)
                delete!(nu_h, n+p_c-1)
                p_c -= 1
            end

            # 3. normalization
            #= The following line tests if the value can be computed (i.e. sufficiently many
            input values are given). Otherwise the algorithm stops. =#
            if !test_bilin(psi_h[n], phi_h[n])
                throw(PolynomialLanczosError("polynomial degrees are too large.", 
                    length(psi_h[n]), length(phi_h[n])))
            end
            if T<:Rational
                U[n+1, n-m_c+m+1] = 1
                V[n+1, n-p_c+p+1] = 1
            else
                U[n+1, n-m_c+m+1] = sqrt(abs(bilin(psi_h[n], phi_h[n])))
                V[n+1, n-p_c+p+1] = U[n+1, n-m_c+m+1]
            end

            phi[n] = phi_h[n] / U[n+1, n-m_c+m+1]
            mu[n] = mu_h[n]
            psi[n] = psi_h[n] / V[n+1, n-p_c+p+1]
            nu[n] = nu_h[n]

            Phi[ell] = [[Phi[ell]; zeros(T, length(phi[n])-size(Phi[ell], 1), size(Phi[ell], 2))] phi[n]]
            Psi[ell] = [[Psi[ell]; zeros(T, length(psi[n])-size(Psi[ell], 1), size(Psi[ell], 2))] psi[n]]

            # 4. compute Delta
            Delta[ell] = bilin(Psi[ell], Phi[ell])
            
            # 5. complete cluster
            if cond(Float64.(Delta[ell])) < 1/dtol
                for i=n+1:n+m_c-1
                    if test_bilin(Psi[ell], reshape(phi_h[i], :, 1))
                        U[(n_[ell]:n).+1, i-m_c+m+1] = Delta[ell] \ bilin(Psi[ell], reshape(phi_h[i], :, 1))
                        phi_h[i] -= [sum(eachcol(Phi[ell]).*U[(n_[ell]:n).+1, i-m_c+m+1]); 
                            zeros(T, length(phi_h[i])-size(Phi[ell], 1))]
                    else
                        delete!.([phi_h], i:n+m_c-1)
                        m_c -= length(i:n+m_c-1)
                        break
                    end
                end
                for i=n+1:n+p_c-1
                    if test_bilin(reshape(psi_h[i], :, 1), Phi[ell])
                        V[(n_[ell]:n).+1, i-p_c+p+1] = (Delta[ell]' \ 
                            bilin(reshape(psi_h[i], :, 1), Phi[ell])')
                        psi_h[i] -= [sum(eachcol(Psi[ell]).*V[(n_[ell]:n).+1, i-p_c+p+1]); 
                            zeros(T, length(psi_h[i])-size(Psi[ell], 1))]
                    else
                        delete!.([psi_h], i:n+p_c-1)
                        p_c -= length(i:n+p_c-1)
                        break
                    end
                end
                if m_c == 0 || p_c == 0
                    warn_orthogonality(bilin, Psi, Phi, orthogonality_tol)
                    return Phi, Psi, U, V, Delta, bilin, n_, mu, nu, D_phi, D_psi, phi_defl, psi_defl
                end
                if mu[n_[ell]] >= m
                    ell_phi = gamma(findkey(mu, mu[n_[ell]]-m))
                end
                if nu[n_[ell]] >= p
                    ell_psi = gamma(findkey(nu, nu[n_[ell]]-p))
                end
                ell += 1
                n_[ell] = n+1
                Phi[ell] = zeros(T, 0, 0)
                Psi[ell] = zeros(T, 0, 0)
            else
                if warn
                    println("Look-ahead step necessary at step $n (starting from 0)")
                end
            end

            # 6. new right polynomial phi_h
            phi_h[n+m_c] = [zeros(T, m); phi[n]]
            mu_h[n+m_c] = mu[n]+m

            I_psi = vcat(filter(x -> x<ell_psi, D_psi), ell_psi:ell-1)
            for k in I_psi
                if test_bilin(Psi[k], reshape(phi_h[n+m_c], :, 1))
                    U[(n_[k]:n_[k+1]-1).+1, n+m+1] = Delta[k] \ bilin(Psi[k], reshape(phi_h[n+m_c], :, 1))
                    phi_h[n+m_c] -= [sum(eachcol(Phi[k]).*U[(n_[k]:n_[k+1]-1).+1, n+m+1]); 
                        zeros(T, length(phi_h[n+m_c])-size(Phi[k], 1))]
                else
                    delete!(phi_h, n+m_c)
                    m_c -= 1
                    break
                end
            end

            # 7. new left polynomial psi_h
            psi_h[n+p_c] = [zeros(T, p); psi[n]]
            nu_h[n+p_c] = nu[n]+p

            I_phi = vcat(filter(x -> x<ell_phi, D_phi), ell_phi:ell-1)
            for k in I_phi
                if test_bilin(reshape(psi_h[n+p_c], :, 1), Phi[k])
                    V[(n_[k]:n_[k+1]-1).+1, n+p+1] = Delta[k]' \ bilin(reshape(psi_h[n+p_c], :, 1), Phi[k])'
                    psi_h[n+p_c] -= [sum(eachcol(Psi[k]).*V[(n_[k]:n_[k+1]-1).+1, n+p+1]); 
                        zeros(T, length(psi_h[n+p_c])-size(Psi[k], 1))]
                else
                    delete!(psi_h, n+p_c)
                    p_c -= 1
                    break
                end
            end

            if m_c == 0 || p_c == 0
                warn_orthogonality(bilin, Psi, Phi, orthogonality_tol)
                return Phi, Psi, U, V, Delta, bilin, n_, mu, nu, D_phi, D_psi, phi_defl, psi_defl
            end
        end
    catch e
        if isa(e, PolynomialLanczosError)
            warn_orthogonality(bilin, Psi, Phi, orthogonality_tol)
            return Phi, Psi, U, V, Delta, bilin, n_, mu, nu, D_phi, D_psi, phi_defl, psi_defl
        else
            rethrow()
        end
    end
end

function warn_orthogonality(bilin, Psi, Phi, tol::Real=1e-8)
    gram_matrix = check_orthogonality(bilin, Psi, Phi)
    if maximum(abs.(I-abs.(gram_matrix))) > tol
        println("Maximum error of Gram matrix: $(maximum(abs.(I-abs.(gram_matrix))))")
    end
end

check_orthogonality(M::Array{T, 3}, p, q) where T <: Real = 
    check_orthogonality(poly_bilin_form(M)[1], p, q)
function check_orthogonality(bilin::Base.Callable, p::Dict{Int, <:AbstractMatrix{T}}, 
        q::Dict{Int, <:AbstractMatrix{T}}) where T <: Real
    p_flat = zeros(T, maximum(x -> size(x, 1), values(p)), sum(x -> size(x, 2), values(p)))
    q_flat = zeros(T, maximum(x -> size(x, 1), values(q)), sum(x -> size(x, 2), values(q)))
    i_ = 1
    for i in sort(collect(keys(p)))
        p_flat[1:size(p[i], 1), i_:i_+size(p[i], 2)-1] = p[i]
        i_ += size(p[i], 2)
    end
    i_ = 1
    for i in sort(collect(keys(q)))
        q_flat[1:size(q[i], 1), i_:i_+size(q[i], 2)-1] = q[i]
        i_ += size(q[i], 2)
    end
    return check_orthogonality(bilin, p_flat, q_flat)
end
function check_orthogonality(bilin::Base.Callable, p::AbstractMatrix{T}, 
        q::AbstractMatrix{T}) where T <: Real
    return [bilin(p[:, i], q[:, j]) for i=1:size(p, 2), j=1:size(q, 2)]
end
