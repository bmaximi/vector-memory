#= This file applies the Lanczos algorithm from lanczos.jl to transform A into an equivalent
banded matrix. This step is optional: A banded matrix A can reduce the computational costs 
compared to a dense matrix; however, as any numerical computations, this step can introduce
further numerical errors. =#

include("lanczos.jl")

# Big floats (256 bit precision) are used since the Lanczos method is quite unstable.
res_lanc = lanczos(big.(A), projection(size(A, 1), d), projection(size(A, 1), d))
A_lanc = Float64.(res_lanc[1][1:size(A, 1), d+1:d+size(A, 1)])
J_lanc = exp(tau*A_lanc)

Sigma0_lanc, G_lanc = lure(A_lanc, d, vacf_0)
#= Since the first entry in vacf is the identity matrix, the following line is more 
complicated than necessary. =#
Sigma_lanc = dcat(vacf[:, :, 1], Sigma0_lanc)

vacf = vacf_lanc = autocorrelation(A_lanc, vacf_0=1)

if showplots
    for i in range2d(d)
        plot!(plots_vacf[i...], plot_x_coords, vacf_lanc[i..., 1:t_step_plot:end])
    end
    display(plot(plots_vacf..., legend=false))
end
