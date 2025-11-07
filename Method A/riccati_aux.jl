include("aux_functions.jl")

#= auxiliary functions for an eigenvalue algorithm for Hamiltonian matrices which respects 
the fact that its eigenvalues are symmetric w.r.t. the imaginary axis according to
    Peter, Benner, Volker Mehrmann, Hongguo Xu
    A numerically stable, structure preserving method for computing the eigenvalues of 
    real Hamiltonian or symplectic pencils, 1998.
=#

function givens(n::Int, i::Int, j::Int, theta::Real)
    # Givens rotation matrix
    @assert 1 <= i <= 2*n
    @assert 1 <= j <= 2*n
    J = sparse(1.0*I(2*n))
    J[i, i] = J[i, i] = cos(theta)
    J[i, j] = sin(theta)
    J[j, i] = -sin(theta)
    return J
end
rotmat(theta) = [cos(theta) sin(theta); -sin(theta) cos(theta)]

function givens_symplectic(n::Int, i::Int, theta::Real)
    return givens(n, i, n+i, theta)
end

function givens_multl!(A::AbstractMatrix{<:Real}, i::Int, j::Int, theta::Real)
    #= In-place multiplication with a Givens matrix (from givens) from the left =#
    n = size(A, 1)
    @assert 1 <= i <= n && 1 <= j <= n && i != j
    A[[i, j], :] = rotmat(theta) * A[[i, j], :]
    return A
end
function givens_symplectic_multl!(A::AbstractMatrix{<:Real}, i::Int, theta::Real)
    n = size(A, 1) ÷ 2
    return givens_multl!(A, i, n+i, theta)
end
function givens_multr!(A::AbstractMatrix{<:Real}, i::Int, j::Int, theta::Real)
    #= In-place multiplication with a symplectic Givens matrix (from givens_symplectic) 
    from the right =#
    n = size(A, 1)
    @assert 1 <= i <= n && 1 <= j <= n && i != j
    A[:, [i, j]] = A[:, [i, j]] * rotmat(theta)
    return A
end
function givens_symplectic_multr!(A::AbstractMatrix{<:Real}, i::Int, theta::Real)
    n = size(A, 1) ÷ 2
    return givens_multr!(A, i, n+i, theta)
end

function householder(v::AbstractVector{<:Real})
    n = length(v)
    return I(n) - 2*v*v' / norm2(v)
end

function householder_symplectic(v::AbstractVector{<:Real})
    P = householder(v)
    return [P 0*I; 0*I P]
end
function householder_multl!(A::AbstractMatrix{<:Real}, v::AbstractVector{<:Real}, factor::Real=2)
    #= In-place multiplication with a Householder matrix (from householder) from the left =#
    v_scaled = factor / norm2(v) * v
    A[:, :] -= v_scaled * (v' * A[:, :])
    return A
end
function householder_multr!(A::AbstractMatrix{<:Real}, v::AbstractVector{<:Real}, factor::Real=2)
    #= In-place multiplication with a Householder matrix (from householder) from the right =#
    v_scaled = factor / norm2(v) * v'
    A[:, :] -= (A * v) * v_scaled
    return A
end

function householder_symplectic_multl!(A::AbstractMatrix{<:Real}, k::Int, v::AbstractVector{<:Real})
    #= In-place multiplication with a symplectic Householder matrix 
    (from householder_symplectic) from the left =#
    n = size(A, 1) ÷ 2
    @assert size(A, 1) == size(A, 2) == 2*n
    @assert all(v[1:k-1] .== 0)
    @assert length(v) == n
    householder_multl!((@view A[1:n, 1:n]), v)
    householder_multl!((@view A[1:n, n+1:2*n]), v)
    householder_multl!((@view A[n+1:2*n, 1:n]), v)
    householder_multl!((@view A[n+1:2*n, n+1:2*n]), v)
    return A
end

function householder_symplectic_multr!(A::AbstractMatrix{<:Real}, k::Int, v::AbstractVector{<:Real})
    #= In-place multiplication with a symplectic Householder matrix 
    (from householder_symplectic) from the right =#
    n = size(A, 1) ÷ 2
    @assert size(A, 1) == size(A, 2) == 2*n
    @assert all(v[1:k-1] .== 0)
    @assert length(v) == n
    householder_multr!((@view A[1:n, 1:n]), v)
    householder_multr!((@view A[1:n, n+1:2*n]), v)
    householder_multr!((@view A[n+1:2*n, 1:n]), v)
    householder_multr!((@view A[n+1:2*n, n+1:2*n]), v)
    return A
end

function special_hessenberg_transformation(A::AbstractMatrix{T}) where T <: Real
    #= implements Algorithm 4.4 from 
        Peter Benner, Volker Mehrmann, Hongguo Xu:
        A numerically stable, structure preserving method for computing the eigenvalues 
        of real Hamiltonian or symplectic pencils
    =#
    n = size(A, 1) ÷ 2

    function householder_vector(x, k)
        u_1 = x[k]
        u_2 = x[k+1:n]
        if norm2(u_2) == 0
            return [zeros(k-1); u_1; zeros(n-k)]
        else
            temp = u_1^2/norm2(u_2)
            u = [zeros(k-1); u_1; (-temp + sqrt(temp^2 + temp))*u_2]
            return u
        end
    end

    A = copy(A)
    Q1 = Matrix{T}(I(2*n))
    Q2 = Matrix{T}(I(2*n))
    for k=1:n-1
        u = householder_vector(A[n+1:end, k], k)
        householder_symplectic_multl!(A, k, u)
        householder_symplectic_multr!(Q1, k, u)

        theta = -atan(A[n+k, k] / A[k, k])
        givens_symplectic_multl!(A, k, -theta)
        givens_symplectic_multr!(Q1, k, theta)

        u = householder_vector(A[1:n, k], k)
        householder_symplectic_multl!(A, k, u)
        householder_symplectic_multr!(Q1, k, u)

        u = householder_vector(A[n+k, 1:n], k+1)
        householder_symplectic_multr!(A, k+1, u)
        householder_symplectic_multr!(Q2, k+1, u)

        theta = atan(A[n+k, k+1] / A[n+k, n+k+1])
        givens_symplectic_multr!(A, k+1, theta)
        givens_symplectic_multr!(Q2, k+1, theta)

        u = householder_vector(A[n+k, n+1:end], k+1)
        householder_symplectic_multr!(A, k+1, u)
        householder_symplectic_multr!(Q2, k+1, u)
    end
    theta = -atan(A[2*n, n] / A[n, n])
    givens_symplectic_multl!(A, n, -theta)
    givens_symplectic_multr!(Q1, n, theta)

    return A, Q1, Q2
end
