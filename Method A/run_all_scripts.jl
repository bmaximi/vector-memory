# This file runs all the other scripts in the correct order.

function run_scripts(scripts)
    for i in scripts
        try
            include(i)
        catch e
            printstyled("Error in file $i: $(e.error).", color=:red)
            println()
        end
        println()
    end
end


run_scripts(["read_input_data.jl", 
    "run_polynomial_lanczos.jl", 
    "run_fix_spurious_negative_eigenvalues.jl", 
    "run_remove_near_spurious_eigenvalues.jl", 
    "run_lanczos.jl"])

