# This file applies the Lanczos algorithm implemented in polynomial_lanczos_method.jl to the data.

include("riccati.jl")
include("polynomial_lanczos_method.jl")
include("get_autocorrelation_function.jl")

# apply the Lanczos method to the data
r = vacf[:, :, 1:t_step:t_step*n0]
# Using higher precision might be necessary for large matrices.
# r = BigFloat.(r)

res = polynomial_lanczos_method(r)
J_start = Matrix(res[4][:, 1+d:end]')

# A_start can be non-real since J_start can have negative real eigenvalues.
A_start = log(J_start) / tau
N = N_start = size(A_start, 1)

if warn && maximum(real(eigvals(A_start))) >= 0
    println("Matrix A is not stable.")
    println("Eigenvalues of A: ", fc(eigvals(A_start)))
end

J, A = J_start, A_start;
