last tested with Julia version 1.11.1

required packages: LinearAlgebra, Plots, DelimitedFiles, SparseArrays, Format, Printf, LaTeXStrings
and backends GR (sketch plots) and PGFPlotsX (nice-looking plots) for the Plots packages if plots are desired.

All files starting with "run_..." and the file "read_input_data.jl" are actual scripts.
To run the program, run the scripts 
- "read_input_data.jl", 
- "run_polynomial_lanczos.jl", 
- "run_fix_spurious_negative_eigenvalues.jl", 
- "run_remove_near_spurious_eigenvalues.jl", 
- "run_lanczos.jl"
in this order or simply run "run_all_scripts.jl" to run them all. 
The other files implement the functions used by these scripts.
To save the results to a text file, call
  save_vacf_data(filename, vacf)
(function from aux_functions.jl) with an arbitrary filename; vacf stores the computed VACF approximation after the scripts have been run.
In the file "read_input_data.jl" you can adjust the calues of n0 (number of points in the grid used for interpolation) and t_step (grid step size) in order to try different grids.

To suppress most output, set warn = false at the beginning of the file aux_functions.jl.
To not display any sketch plots while the scripts are run, set showplots = false there.

The files starting with "plot_..." create the two plots. They can be run independently provided that the files in "Input_data" and "results" are present. The plots are displayed and the corresponding tex code is saved to a file (to be compiled using LuaLaTeX).
