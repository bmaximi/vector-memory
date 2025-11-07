
#= AutosizeArray implements an array whose size increases automatically when an entry is set
whose indices are larger than its current size. The new entries are set to 0 if they 
are not explicitly set otherwise.
Indices must still be positive integers. =#

mutable struct AutosizeArray{T, N} <: AbstractArray{T, N}
    #= size denotes the "actual" size of the object. This size can be smaller than the size
    of the used array. =#
    size::NTuple{N, Int}
    base::Array{T, N}
    # The last type parameter L is not specified since true and fals are possible.
    view::SubArray{T, N, Array{T, N}, NTuple{N, UnitRange{Int64}}}
    function AutosizeArray{T, N}(A::AbstractArray{T, N}) where {T, N}
        base = Array(A)
        return new{T, N}(size(A), base, view(base, (1:size(base, i) for i=1:N)...))
    end
end
AutosizeArray(T::Type, N::Int) = AutosizeArray{T, N}()
AutosizeArray{T, N}() where {T, N} = AutosizeArray(zeros(T, repeat([0], N)...))
AutosizeArray(A::AbstractArray{T, N}) where {T, N} = AutosizeArray{T, N}(A)
const AutosizeVector{T} = AutosizeArray{T, 1}
AutosizeVector(A::AbstractVector{T}) where T = AutosizeVector{T}(A)
# AutosizeVector{T}(x...) where T = AutosizeArray{T, 1}(x...)
const AutosizeMatrix{T} = AutosizeArray{T, 2}
AutosizeMatrix(A::AbstractMatrix{T}) where T = AutosizeMatrix{T}(A)
# AutosizeMatrix{T}(x...) where T = AutosizeArray{T, 2}(x...)
size(A::AutosizeArray) = A.size
size(A::AutosizeArray, dim::Integer) = A.size[dim]
function clean!(A::AutosizeArray{T, N})::AutosizeArray{T, N} where {T, N}
    A.base = Array(A)
    A.view = view(A.base, (1:A.size[i] for i=1:N)...)
    return A
end

convert(::Type{AutosizeArray{T, N}}, A::AbstractArray{T, N}) where {T, N} = AutosizeArray{T, N}(A)
convert(::Type{AutosizeArray{T, N}}, A::AutosizeArray{T, N}) where {T, N} = A
copy(A::AutosizeArray{T, N}) where {T, N} = AutosizeArray{T, N}(A.view)

Array{T, N}(A::AutosizeArray{T, N}) where {T, N} = A.base[(1:A.size[i] for i=1:N)...]
getindex(A::AutosizeArray{T, N}, i...) where {T, N} = getindex(A.view, i...)
# Base.isassigned(A::AutosizeArray{T, N}, i) where {T, N} = isassigned(A.view, i)
Base.isassigned(A::AutosizeArray{T, N}, i::NTuple{N, Int}) where {T, N} = all(1 .<= i .<= size(A))::Bool
Base.isassigned(A::AutosizeArray{T, N}, i::CartesianIndex{N}) where {T, N} = Base.isassigned(A, Tuple(i))::Bool
Base.isassigned(A::AutosizeArray{T, N}, i::Int) where {T, N} = 1 <= i <= length(A)::Bool

function setindex!(A::AutosizeArray{T, N}, x, index::Vararg{Union{Int, AbstractRange}, N}; 
        extend=true)::AutosizeArray{T, N} where {T, N}
    # new_base_size = collect(size(A.base))
    new_base_size = Vector{Int}()
    new_size = collect(A.size)
    change_size = false
    replace_array = false
    for i=1:N
        if minimum(index[i]) <= 0
            throw(BoundsError(A, index))
        elseif maximum(index[i]) > new_size[i]
            change_size = true
            new_size[i] = maximum(index[i])
            if new_size[i] > size(A.base, i)
                if !extend
                    throw(BoundsError(A, index))
                end
                if !replace_array
                    # new_base_size is initialized only when it is used.
                    new_base_size = collect(size(A.base))
                end
                new_base_size[i] = max(new_size[i], 2*A.size[i])
                replace_array = true
            end
        end
    end
    if replace_array
        old_base = A.base
        A.base = zeros(T, new_base_size...)
        A.base[axes(A)...] = A.view
        A.size = ntuple(i -> new_size[i], N)
        A.view = view(A.base, (1:new_size[i] for i=1:N)...)
    end
    A.base[index...] = x
    if change_size
        A.size = ntuple(i -> new_size[i], N)
        A.view = view(A.base, (1:new_size[i] for i=1:N)...)
    end
    return A
end
