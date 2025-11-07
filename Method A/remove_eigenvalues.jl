#= This file implements the functions used in run_fix_bad_eigenvalues.jl and 
run_remove_near_spurious_eigenvalues.jl to remove and duplicate eigenvalues. =#


function matrix_from_eigen(vals::Vector{T}, vecs::Matrix{T})::Matrix{T} where T<:Complex
    return vecs * diagm(vals) / vecs
end

function prony_coefficients(A::AbstractMatrix{<:Real}, d::Integer=d; kw...)
        n = size(A, 1)
    return prony_coefficients(A, projection(n, d), projection(n, d), d; kw...)
end
function prony_coefficients(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real}, 
        C::AbstractMatrix{<:Real}, d::Integer=d; 
        A_vecs::Union{Nothing, Matrix{<:Number}}=nothing)
    # computes and returns the coefficients of the Prony series B'*exp(t*A)*C
    if A_vecs === nothing
        A_vecs = eigvecs(A)
    end
    y = reshape(A_vecs' * B, :, d, 1)
    z = reshape(A_vecs \ C, :, 1, d)
    #= Note: Due to rounding errors, the coefficient in omega corresponding to a complex 
    conjugated eigenvalue is not always exactly the complex conjugated coefficient and the 
    imaginary part of a coefficient corresponding to a real eigenvalue is no always 0. =#
    omega = conj(y) .* z
    return omega
end

function keep_eigenvalues(eigvals, eigvecs, inds::BitVector, d; kw...)
    return keep_eigenvalues(eigvals, eigvecs, findall(inds), d; kw...)
end
function keep_eigenvalues(eigvals::Vector{<:Number}, eigvecs::Matrix{<:Number}, 
        inds::Vector{<:Integer}, d::Integer; tau::Real=tau, tol_imag::Real=tol_imag, 
        warn::Bool=warn)
    #= inds contains the indices of the eigenvalues to keep; all other eigenvalues and the 
    corresponding eigenvectors (and as many rows in the remaining eigenvectors) are removed;
    returns the remaining eigenvalues and eigenvectors and the reassembled matrices J and
    A = 1/tau*log(J). =#
    N = length(eigvals)
    k = length(inds)
    eigvals_res = eigvals[inds]
    eigvecs_res = [eigvecs[1:d, inds]; zeros(k-d, k)]
    #= The first d rows of eigvecs are copied since they correspond to the d velocity
    components. After that, the row whose (normed) orthogonal projection to the linear hull
    of all rows copied so far has the largest norm is copied. In this way no rows which are 
    (nearly) linearly dependant of the previous rows are taken. The order of the rows is 
    not retained except for the first d rows. =#
    eigvecs_proj = eigvecs[:, inds]
    for i=1:d
        new_row = eigvecs[i, inds]
        new_norm = new_row'*new_row
        for j=i:N
            # Orthogonalize all other rows against the i-th row.
            eigvecs_proj[j, :] -= (new_row'*eigvecs_proj[j, :]) / new_norm * new_row
        end
    end
    deleted_inds = Set(d+1:N)
    for i=d+1:k
        new_ind = maximum(x -> (norm(eigvecs_proj[x, :]), x), deleted_inds)[2]
        new_row = eigvecs_proj[new_ind, :]
        new_norm = new_row'*new_row
        eigvecs_res[i, :] = eigvecs[new_ind, inds]
        pop!(deleted_inds, new_ind)
        for j in deleted_inds
            # Orthogonalize all other rows against the new row.
            eigvecs_proj[j, :] -= (new_row'*eigvecs_proj[j, :]) / new_norm * new_row
        end
    end
    eigvals_log = log.(eigvals_res) / tau
    J = matrix_from_eigen(eigvals_res, eigvecs_res)
    # J should already be real, hence the following line only corrects rounding errors.
    J_real = assert_real(J, "J", tol=tol_imag, warn=warn)
    A = matrix_from_eigen(eigvals_log, eigvecs_res)
    A_real = assert_real(A, "A", tol=tol_imag, warn=warn)

    return eigvals_res, eigvecs_res, eigvals_log, J_real, A_real
end


function duplicate_negative_eigenvalues(eigvals::Vector{<:Number}, eigvecs::Matrix{<:Number}, 
        omega::Union{Nothing, Array{<:Number, 3}}=nothing)
    inds = findall((imag(eigvals).==0) .& (real(eigvals).<=0))
    return duplicate_eigenvalues(eigvals, eigvecs, inds, omega)
end
function duplicate_eigenvalues(eigvals::Vector{T}, eigvecs::Matrix{T}, inds::Vector{<:Integer}, 
        omega::Union{Nothing, Array{T, 3}}=nothing) where T <: Number
    #= returns the eigenvalues and eigenvectors of the matrices J and A = 1/tau*log(J) where 
    negative real eigenvalues have been doubled; the two logarithms of these doubled eigenvalues
    are complex conjugate numbers. =#

    k = length(eigvals)
    d = size(omega, 2)
    eigvals_res = [eigvals; zeros(complex(T), length(inds))]
    eigvals_log = [log.(complex(eigvals)); zeros(complex(T), length(inds))]
    eigvecs_res = [eigvecs zeros(complex(T), k, length(inds)); zeros(length(inds), k+length(inds))]
    if omega !== nothing
        omega_res = [omega; zeros(complex(T), length(inds), d, d)]
    end
    for (i, ind) = enumerate(inds)
        #= Since +0.0im and -0.0im are separate complex numbers, the logarithm automatically
        uses the correct sign. =#
        eigvals_res[k+i] = conj(eigvals_res[ind])
        eigvals_log[k+i] = conj(eigvals_log[ind])
        eigvecs_res[k+i, ind] = im
        eigvecs_res[:, k+i] .= conj(eigvecs_res[:, ind])
        if omega !== nothing
            omega_res[ind, :, :] /= 2
            omega_res[k+i, :, :] = conj(omega_res[ind, :, :])
        end
    end
    if omega !== nothing
        return eigvals_res, eigvecs_res, eigvals_log, omega_res
    else
        return eigvals_res, eigvecs_res, eigvals_log
    end
end

function remove_bad_eigenvalues(J::Matrix{<:Real}; tau::Real=tau, warn::Bool=warn,
        d::Integer=d, tol_imag::Real=1e-10)
    #= This function removes spurious eigenvalues and duplicates negative real ones. =#
    J_vals_0, J_vecs_0 = eigen(J)
    omega_0 = prony_coefficients(J, d, A_vecs=J_vecs_0)

    # inds stores the indices of all eigenvalues whose absolute value is smaller than 1
    inds_keep = abs.(J_vals_0) .< 1
    # All eigenvalues with absolute value greater than 1 are removed.
    J_vals, J_vecs, _, _, _ = keep_eigenvalues(J_vals_0, J_vecs_0, inds_keep, d, warn=false)
    if warn && !all(inds_keep)
        println("Removed $(sum(.!inds_keep)) eigenvalues of J: $(fc(J_vals_0[.!inds_keep]))")
        println("Norms of the corresponding Prony coefficients: "*
            "$(fc(norm.(eachslice(omega_0[.!inds_keep, :, :], dims=1)))) "*
            "(Maximum: $(fc(maximum(norm.(eachslice(omega_0[.!inds_keep, :, :], dims=1))))))")
    end
    omega = omega_0[inds_keep, :, :]
    J_vals_res, J_vecs_res, A_vals_res, omega_res = duplicate_negative_eigenvalues(J_vals, J_vecs, omega)
    A_vals_res /= tau

    J_res = matrix_from_eigen(J_vals_res, J_vecs_res)
    # J should already be real, hence the following line only corrects rounding errors.
    J_real = assert_real(J_res, "J", tol=tol_imag, warn=warn)
    A_res = matrix_from_eigen(A_vals_res, J_vecs_res)
    A_real = assert_real(A_res, "A", tol=tol_imag, warn=warn)

    return omega_0, J_vals_res, J_vecs_res, A_vals_res, J_real, A_real, omega_res
end
