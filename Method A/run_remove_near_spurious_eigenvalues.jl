#= Thsi file removes further eigenvalues which are not strictly spurious but whose presence 
prevents the autocorrelation function from being of positive type. It also removes eigenvalues 
with negligible coefficients in the Prony series. =#

include("riccati.jl")
include("remove_eigenvalues.jl")

J_pos = J_fixed
A_pos = A_fixed
N = size(A_pos, 1)

max_eig = maximum(abs.(A_vals))

#= In the following, additional eigenvalues with negative real part are removed if the
Lur'e equations do not have a solution. Out of all eigenvalues at whose imaginary 
(on the imaginary axis) part the Fourier transform is negative, the eigenvalue 
(or the pair of complex conjugated eigenvalues) with the largest (i.e. closest to 0) 
real part is/are removed, until the Lur'e equations have a solution. 
(The weight of the eigenvalues in the Prony series is ignored.) =#
A_vals_order = sortperm(A_vals, by=real)
A_vals_order_inv = invperm(A_vals_order)
inds = trues(N)

#= In the loop condition, the values of the Fourier transform at the imaginary axis are 
computed. If one of them has a negative eigenvalue, a step of the loop is performed. 
The Fourier transform is re-evaluated after each step since it is often not necessary
to remove all eigenvalues where the Fourier transform is not positive semi-definite
at the beginning. =#
while (begin
            vacf_hat = get_vacf_hat(A_pos, d)
            vacf_hat_values = vacf_hat.(imag(A_vals[inds]))
            minimum(minimum.(eigvals.(vacf_hat_values))) < 0
        end)
    global A_pos, J_pos, inds
    println("Fourier transform is negative for $(sum(minimum.(eigvals.(vacf_hat_values)) .< 0)) "*
        "out of $(sum(inds)) remaining eigenvalues")
    inds_neg = falses(N)
    inds_neg[inds] = minimum.(eigvals.(vacf_hat_values)) .< 0
    remove_ind = findlast(inds_neg[A_vals_order])

    inds[A_vals_order[remove_ind]] = false
    if (remove_ind > 1 && 
            abs(A_vals[A_vals_order[remove_ind]] - conj(A_vals[A_vals_order[remove_ind-1]])) 
            / max_eig < 1e-12)
        inds[A_vals_order[remove_ind-1]] = false
    end

    (_, _, _, J_pos, A_pos) = keep_eigenvalues(J_vals, J_vecs, findall(inds), d)
end
J, A = J_pos, A_pos
N = N_pos = size(J_pos, 1)

if warn && N_fixed != N_pos
    temp = sortperm(A_vals[.!inds], by=real)
    println("Removed the following $(N_fixed-N_pos) eigenvalues of A to obtain a "*
        "positive real system: $(fc(A_vals[.!inds][temp]))")
    println("Corresponding Prony coefficients: "* 
        "$(fc(norm.(eachslice(omega_fixed[.!inds, :, :][temp], dims=1))))")
end

# In the following, all eigenvalues whose coefficient in the Prony series is small are removed.
threshold_omega = 1e-6
temp = norm.(eachslice(omega_fixed[inds, :, :], dims=1)) .< threshold_omega
if any(temp)
    if warn
        println("Removed the following $(sum(temp)) eigenvalues of A with coefficient norm "*
            "smaller than $(fc(threshold_omega)): $(fc(A_vals[inds][temp]))")
        println(", Prony coefficients: "*
            "$(fc(norm.(eachslice(omega_fixed[inds, :, :][temp, :, :], dims=1))))")
    end
    inds[norm.(eachslice(omega_fixed, dims=1)) .< threshold_omega] .= false

    (_, _, _, J_pos, A_pos) = keep_eigenvalues(J_vals, J_vecs, findall(inds), d)
else
    if warn
        println("Smallest norm of a remaining Prony coefficient: "*
            "$(fc(minimum(norm.(eachslice(omega_fixed[inds, :, :], dims=1)))))")
    end
end

J, A = J_pos, A_pos
N = N_pos = size(J_pos, 1)

Sigma0_pos, G_pos = lure(A_pos, d, vacf_0)
Sigma_pos = dcat(vacf_start[:, :, 1], Sigma0_pos)

vacf = vacf_pos = autocorrelation(A_pos, vacf_0=1)

vacf_error = sum((vacf-vacf_start).^2, dims=(1, 2))[1, 1, :]
println("Maximum interpolation error (Frobenius norm)...")
temp = findmax(vacf_error[1:t_step:(N_start-1)*t_step+1])
println("...at the interpolation points: $(fc(sqrt(temp[1]))) at $(fc(tau*(temp[2]-1)))")
temp = findmax(vacf_error[1:(N_start-1)*t_step+1])
println("...between the interpolation points: $(fc(sqrt(temp[1]))) at $(fc(delta_t*(temp[2]-1)))")
temp = findmax(vacf_error)
println("...at all data points: $(fc(sqrt(temp[1]))) at $(fc(delta_t*(temp[2]-1)))")
println("Number of auxiliary variables: $(size(A, 1)-d)")

if showplots
    for i in range2d(d)
        plot!(plots_vacf[i...], plot_x_coords, vacf[i..., 1:t_step_plot:end])
    end
    display(plot(plots_vacf..., legend=false, title="n_0=$N_start, tau=$tau"))
end
