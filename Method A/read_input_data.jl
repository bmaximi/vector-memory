# This file reads the data file and initializes some variables.

include("aux_functions.jl")
# plotlyjs()

d = 2

filenames = ["vxvxc" "vxvyc"; "vyvxc" "vyvyc"]
vacf_temp = readdlm.("Input_data/" .* filenames .* ".txt")

# vacf  = velocity autocorrelation (or auto-covariance) function
vacf_input = zeros(size(filenames)..., size(vacf_temp[1], 1))
for i in eachindex(IndexCartesian(), filenames)
    vacf_input[i, :] = vacf_temp[i][:, 2]
end

#= normalize autocorrelation data by dividing by the Cholesky decomposition of its value at 0 
from the left and the right =#
vacf_0 = vacf_input[:, :, 1]
vacf_start = vacf = normalize_input_data(vacf_input)

# delta_t is the grid size of the grid on which the input data is given.
delta_t = vacf_temp[1, 1][2, 1]
n = size(vacf, 3) - 1
#= n0 indicates the number of input points used. Corresponds to 2*n from the paper.
Consequently n0 = 30 produces the results shown in the figures while setting
n0 = 30, 32, 34, ,..., 60 produces the results from the table =#
n0 = 30

#= t_step is the number of time steps between two points in the time grid of the actual 
used data. Each time step has length delta_t. 
tau = delta_t*t_step is the grid size of the grid which is used for the algorithm. =#
t_step = 50
tau = delta_t*t_step

println("n0 = $n0, tau = $tau")

# Plot input data
t_step_plot = 1
y_lims_diff = (maximum(vacf, dims=3) - minimum(vacf, dims=3))[:, :, 1]
plot_y_lims = tuple.(minimum(vacf, dims=3)-0.02*y_lims_diff, 
    maximum(vacf, dims=3)+0.02*y_lims_diff)
plots_vacf = [plot(ylims=plot_y_lims[i, j]) for i=1:d, j=1:d]
plot_x_coords = delta_t * (0:t_step_plot:size(vacf, 3)-1)
if showplots
    for i in range2d(d)
        plot!(plots_vacf[i...], plot_x_coords, vacf[i..., 1:t_step_plot:end])
        plot!(plots_vacf[i...], tau*(0:n0-1), vacf[i..., 1:t_step:t_step*n0], 
            seriestype=:scatter, markersize=2)
    end
    display(plot(plots_vacf..., layout=(d, d), legend=false, title="n_0=$n0, tau=$tau"))
end
