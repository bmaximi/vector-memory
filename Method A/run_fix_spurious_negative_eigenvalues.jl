#= This file removes spurious eigenvalues from the matrices J and A (i.e. eigenvalues of J 
with absolute value >= 1 and eigenvalues of A with non-negative real part) and duplicates 
negative real eigenvalues. =#

include("riccati.jl")
include("remove_eigenvalues.jl")

# Eigenvalues with non-negative real part are removed and negative real one are duplicated.
omega, J_vals, J_vecs, A_vals, J_fixed, A_fixed, omega_fixed = remove_bad_eigenvalues(J, warn=warn)

J, A = J_fixed, A_fixed
N = N_fixed = size(J_fixed, 1)

try
    Sigma0_fixed, G_fixed = lure(A_fixed, d, vacf_0)
    Sigma_fixed = dcat(vacf_start[:, :, 1], Sigma0_fixed)
catch e
    if isa(e, RiccatiError) && warn
        println("Riccati equation has no solution.")
    end
end

vacf = vacf_fixed = autocorrelation(A_fixed, vacf_0=1)

vacf_error = sum((vacf-vacf_start).^2, dims=(1, 2))[1, 1, :]
println("Maximum interpolation error (Frobenius norm)...")
temp = findmax(vacf_error[1:t_step:(n0-1)*t_step+1])
println("...at the interpolation points: $(fc(sqrt(temp[1]))) at $(fc(tau*(temp[2]-1)))")
temp = findmax(vacf_error[1:(n0-1)*t_step+1])
println("...between the interpolation points: $(fc(sqrt(temp[1]))) at $(fc(delta_t*(temp[2]-1)))")
temp = findmax(vacf_error)
println("...at all data points: $(fc(sqrt(temp[1]))) at $(fc(delta_t*(temp[2]-1)))")

if showplots
    for i in range2d(d)
        plot!(plots_vacf[i...], plot_x_coords, vacf_fixed[i..., 1:t_step_plot:end])
    end
    display(plot(plots_vacf..., legend=false, title="n_0=$n0, tau=$tau"))
end
