include("aux_functions.jl")

#= get_autocorrelation_function evaluates the autocorrelation function according to 
the coordinates given by P and the system given by A at the points 
0, tau, ..., n*tau and returns the result =#

function get_autocorrelation_function(A::AbstractMatrix{<:Number}; 
        vacf_0=vacf_0, tau=delta_t, n=n, kw...)
    return get_autocorrelation_function(A, vacf_0, tau, n; kw...)
end
function get_autocorrelation_function(A::AbstractMatrix{T}, vacf_0::Real, tau::Real, 
        n::Integer, P, Sigma::UniformScaling=I) where T <: Real
    return get_autocorrelation_function(A, vacf_0, tau, n, P, Sigma(size(A, 1)))
end
function get_autocorrelation_function(A::AbstractMatrix{T}, vacf_0::Real, tau::Real, 
        n::Integer, P::AbstractVector{<:Real}) where T <: Real
    return get_autocorrelation_function(A, vacf_0, tau, n, reshape(P, :, 1))[1, 1, :]
end
function get_autocorrelation_function(A::AbstractMatrix{T}, vacf_0::Real, tau::Real, 
        n::Integer) where T <: Real
    return get_autocorrelation_function(A, vacf_0, tau, n, projection(size(A, 1), d))
end
function get_autocorrelation_function(A::AbstractMatrix{T}, 
        vacf_0::Real, tau::Real, n::Integer, P::AbstractMatrix{<:Real}, 
        Sigma::AbstractMatrix{<:Real}=I) where T <: Real
    d = size(P, 2)
    exp_tauA = exp(tau*A)
    r = ones(T, d, d, n+1)
    temp = I
    Sigma_P = Sigma*P
    for i = 1:n+1
        r[:, :, i] = vacf_0^2 * P'*temp*Sigma_P
        temp = temp*exp_tauA
    end
    return r
end
autocorrelation = get_autocorrelation_function
