# last tested with Julia version 1.11.1

# This file contains several auxiliary functions used by other files.

using DelimitedFiles
using SparseArrays
using LinearAlgebra
using Plots
using Format
using Printf
using LaTeXStrings
import Base.size, Base.getindex, Base.setindex!, Base.convert, Base.copy, Base.Callable, Base.isexpr
include("autosizearray.jl")
gr()
# plotlyjs()
# pgfplotsx()
showplots = true
warn = true


function normalize_input_data(vacf_input::Array{<:Real, 3})
    vacf = zeros(size(vacf_input)...)
    L0 = cholesky(vacf_input[:, :, 1]).L
    for i = axes(vacf_input, 3)
        vacf[:, :, i] = L0 \ vacf_input[:, :, i] / L0'
    end
    vacf[:, :, 1] = I(size(vacf_input, 1))
    return vacf
end

BigRational = Rational{BigInt}
norm2(x) = x'*x
range2d(d) = Iterators.product(1:d, 1:d)
isstable(A::AbstractMatrix{<:Number}) = maximum(real(eigvals(A))) < 0
maximag(x::AbstractArray{<:Number}) = maximum(abs.(imag.(x)))

function projection(T::Type, rows::Integer, cols::Integer)
    if cols > rows
        error("cols > rows in projection matrix")
    end
    return [I(cols); zeros(T, rows-cols, cols)]
end
projection(rows::Integer, cols::Integer) = projection(Bool, rows, cols)

tol_imag = 1e-10
function assert_real(x::AbstractArray{<:Number}, name::String; tol::Real=tol_imag, 
        warn::Bool=warn)
    if warn && maximag(x) > tol
        println("$name is not real. Maximum imaginary part: $(fc(maximag(x)))")
    end
    return real(x)
end

function get_vacf_hat_im(A::AbstractMatrix{<:Real}, d::Integer, 
        vacf_0::Union{Real, AbstractMatrix{<:Real}}=vacf_0)::Function
    #= returns a function which maps xi to the Fourier transform of the velocity autocorrelation 
    function at the point xi*(-im); the result is real if xi is real. =#
    vacf_hat(xi::Number) = begin
        temp = inv(xi*I-A)
        return temp[1:d, 1:d]*vacf_0 + vacf_0*temp[1:d, 1:d]'
    end
    return vacf_hat
end
function get_vacf_hat(A::AbstractMatrix{<:Real}, d::Integer, 
        vacf_0::Union{Real, AbstractMatrix{<:Real}}=vacf_0)::Function
    vacf_hat_im = get_vacf_hat_im(A, d, vacf_0)
    vacf_hat(xi) = vacf_hat_im(xi*im)
    return vacf_hat
end

dcat(M::Any...) = cat(M..., dims=(1, 2))::AbstractMatrix


function save_vacf_data(filename::String, data::AbstractArray{<:Real, 3})
    vacf_matrix = reshape(permutedims(data, (2, 1, 3)), 4, :)'
    vacf_string = Printf.format.([Printf.Format("%1.7e")], vacf_matrix)
    writedlm(filename, vacf_string)
end


#= format_complex formats a complex number. The real and imaginary parts of a complex numver
are formatted with Printf.format separately. Real and imaginary parts which are 0 are not
displayed. =#
function format_complex(z::Number; spec="%g")
    if imag(z) == 0
        return Printf.format(Printf.Format(spec), real(z))
    elseif real(z) == 0
        return Printf.format(Printf.Format(spec*"im"), imag(z))
    elseif imag(z) < 0
        return Printf.format(Printf.Format(spec*"-"*spec*"im"), real(z), -imag(z))
    else
        return Printf.format(Printf.Format(spec*"+"*spec*"im"), real(z), imag(z))
    end
end

format_complex(z::AbstractVector{T}; spec="%g", sq_brackets::Bool=false, sep=", ") where T <: Number = 
    sq_brackets ? "[$(join(format_complex.(z, spec=spec), sep))]" : join(format_complex.(z, spec=spec), sep)

format_complex(z::AbstractMatrix{T}; spec="%g", colsep=" ", rowsep="; ") where T <: Number =
    "[$(join(format_complex.(eachrow(z), spec=spec, sq_brackets=false, sep=colsep), rowsep))]"
fc = format_complex


# auxiliary functions for plots

function format_axis(number::Real, post_digits::Int, total_digits::Int, minus_sign::Bool=true)
    #= auxiliary script for padding a number by visible zeros from the right until it
    has num_digits digits and invisible zeros from the left until it has width digits;
    useful as yformatter argument to plot for aligning y-axis labels in subplots by
    making all ticks have equal length
    currently not adapted to numbers with exponential part =#
    result = format(abs(number), precision=post_digits)
    digits = length.(split(result, "."))
    # add (visible or invisible) comma
    if minus_sign
        if number < 0
            result = "-" * result
        else
            result = "\\phantom-" * result
        end
    end
    result = "{$result}"
    if digits[1]+post_digits < total_digits
        # pad from the left width invisible zeros
        result = "\\phantom{$(0^(total_digits-digits[1]+digits[2]))}" * result
    end
    return latexstring(result)
end

function save_plot(plot::Plots.Plot, output_path::Tuple{<:AbstractString, <:AbstractString}; 
        tex_version::Union{Bool, Symbol}=true, text_replace::AbstractVector{<:Tuple}=Tuple[])
    # several post-processing steps are applied to the plot's tex code, then it is saved to a file.

    io = IOBuffer()
    Plots.tex(plot, io)

    # TeXStudio comments
    if tex_version === true || tex_version === :pdftex
        tex_command = "pdflatex"
    elseif tex_version === :luatex
        tex_command = "lualatex"
    else
        error("Unknown option for \"tex_version\"")
    end
    text = """% !TeX encoding = UTF-8
    % !TeX program = $tex_command
    """ * String(take!(io))

    # replace space by tabs
    text = replace(text, "    "=>"\t")
    # keep at most 6 significant digits of the plot data
    text = replace(text, r"(\t\t\t[0-9]+\.[0-9]*  )(-?[1-9][0-9]*\.|-?0\.0*[1-9])([0-9]{5})([0-9]*)(e-?[1-9][0-9]*)?  \\\\" => s"\1\2\3\5 \\\\")
    # correct rounding error with many zeros or nines in tikz parameters
    text = replace(text, r"0{10}[0-9]{1,2}([^0-9])" => s"\1")
    for i=0:8
        text = replace(text, Regex(string(i)*"9{10,}[0-9]{1,2}([^0-9])") => SubstitutionString(string(i+1)*s"\1"))
    end
    #= The width of plots with inset plots (lens!) is 5pt smaller than required. This bug can 
    be corrected here. For this one has to determine the actual width from the tex file and 
    replace it here by another value =#
    for i in text_replace
        if !occursin(i[1], text)
            println("Warning: Expression to be replaced is not found: $(i[1])")
        end
        text = replace(text, i[1]=>i[2])
    end

    write(joinpath(output_path[1], "$(output_path[2]).tex"), text)
end
