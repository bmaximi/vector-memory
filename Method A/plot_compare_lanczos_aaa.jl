
include("aux_functions.jl")

pgfplotsx()
#= For faster display, use the gr backend instead of pfgplotsx, but LaTeX code in labels will 
not be displayed correctly. =#
# gr()

d = 2

input_files = ["vxvxc" "vxvyc"; "vyvxc" "vyvyc"]
vacf_temp = readdlm.("Input_data/" .* input_files .* ".txt", header=false)
delta_t = vacf_temp[1, 1][2, 1]

vacf_input = zeros(size(input_files)..., size(vacf_temp[1], 1))
for i in eachindex(IndexCartesian(), input_files)
    vacf_input[i, :] = vacf_temp[i][:, 2]
end
# normalize autocorrelation
for i = 2:size(vacf_input, 3)
    vacf_input[:, :, i] = vacf_input[:, :, i] / vacf_input[:, :, 1]
end
vacf_input[:, :, 1] = I(d)
n = size(vacf_input, 3) - 1

# The input data resolution is very high, so only every 10th input data point is plotted.
t_step_plot = 10

# Create plots and set plot parameters
plot_legend = [:topright false; false false]
plot_yticks = [-0.2:0.2:1.0  -0.02:0.02:0.1;  -0.02:0.02:0.1  -0.2:0.2:1.0]

max_digits = [1 2; 2 1]

plot_x_coords = delta_t * (0:t_step_plot:n)
plots = [plot(xlims=(-0.5, 50.05), legend=plot_legend[i, j], legend_font_halign=:left, 
        top_margin=(-4, :mm), right_margin=(3, :mm), bottom_margin=(-1, :mm),
        xlabel=L"$t$", ylabel=L"$C_{Y_{%$i}, Y_{%$j}}(t)$",
        yformatter=x->format_axis(x, max_digits[i, j], maximum(max_digits)+1),
        ytricks=plot_yticks[i, j])
    for (i, j) in range2d(d)
]

# Plot input vacf
for k in range2d(d)
    plot!(plots[k...], [0], [0], color=:black, style=:dash, label="VACF")
    plot!(plots[k...], plot_x_coords, vacf_input[k..., 1:t_step_plot:end], color=:black, 
        style=:dash, label="")
end

# Method A
input_path_A = "results/vacf_method_A.txt"
temp = readdlm(input_path_A)
vacf_A = zeros(d, d, size(temp, 1))
for (i, k) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)])
    vacf_A[k..., :] = temp[:, i]
end

for k in range2d(d)
    plot!(plots[k...], plot_x_coords, vacf_A[k..., 1:t_step_plot:end], color=1,
        label=L"Method A ($N=16$)")
end

# Method B
input_path_B = "results/vacf_method_B.txt"
temp = readdlm(input_path_B)
vacf_B = zeros(d, d, size(temp, 1))
for (i, k) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)])
    vacf_B[k..., :] = temp[:, i]
end

for k in range2d(d)
    plot!(plots[k...], plot_x_coords, vacf_B[k..., 1:t_step_plot:end], color=2, 
        label=L"Method B ($N=12$)")
end

# Create inset plots
for k in range2d(d)
    if k == (1, 1)
        lens!(plots[k...], [16.9, 50.05], [-0.0035, 0.0012], 
            inset=(1, bbox(0.32, 0.28, 0.7, 0.48)), 
            top_margin=(-3, :mm), bottom_margin=(-2, :mm), linewidth=0)
    elseif k == (2, 2)
        lens!(plots[k...], [16.9, 50.05], [-0.0073, 0.0069], 
            inset=(1, bbox(0.32, 0.01, 0.7, 0.60)), 
            top_margin=(-3, :mm), bottom_margin=(-2, :mm), linewidth=0)
    else
        lens!(plots[k...], [16.9, 50.05], [-0.0021, 0.0014], 
            inset=(1, bbox(0.32, 0.01, 0.7, 0.60)), 
            top_margin=(-3, :mm), bottom_margin=(-2, :mm), linewidth=0)
    end
end

t_step = 50
tau = delta_t*t_step
n0 = 30
r = vacf_input[:, :, 1:t_step:t_step*n0]
# Plot the grid as black dots
for k in range2d(d)
    plot!(plots[k...], tau*(0:n0-1), r[k..., :], color=:black, 
        seriestype=:scatter, markersize=1.5, xticks=:auto, markerstrokewidth=0.3, primary=false)
end

plot_size=(630, 420)
p = plot(plots..., layout=(d, d), size=plot_size, tex_output_standalone=true)
display(p)
#= fix width (bug where lens! reduces the plot width) and make lens frame invisible =#
text_replace = [("width={71.01mm}", "width={76.01mm}"), 
    (r"(\\addplot\[color=\{rgb,1:red,0\.8275;green,0\.8275;blue,0\.8275\}, name path=\{[0-9]+\}, draw opacity=\{)1\.0(\}, line width=\{0\}, solid(?:, forget plot)?\])",
    s"\1 0.0\2")]
output_path = ("", "compare_lanczos_aaa")

# manuelle Nachbearbeitung des LaTeX-Codes und Speichern
save_plot(p, output_path, text_replace=text_replace, tex_version=:luatex)
