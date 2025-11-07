#= This file implements the Lanczos algorithm for Lanczos vectors used in run_lanczos.jl.
Although it is related to the Lanczos algorithm with orthogonal polynomials, it is not the
same algorithm. =#

include("aux_functions.jl")

function lanczos(A::AbstractMatrix{T}, L::AbstractMatrix{<:Real}, R::AbstractMatrix{<:Real}; 
        dtol=T == BigFloat ? 1e-50 : 1e-6, stop_at=100) where T <: Real
    #= Lanczos algorithm according to
        Aliaga, Boley, Freund, Hernandez:
        A Lanczos-type method for multiple starting vectors (1999)
    Notation is mostly taken from there. Indices are indicated by a prefixed underscore at 
    the end of the variable name.
    
    The starting vectors are given as columns of the matrices L (left starting vectors)
    and R (right starting vectors). =#
    m = size(R, 2)
    p = size(L, 2)
    N = size(A, 1)
    @assert N == size(A, 2) == size(L, 1) == size(R, 1)
    norm_A = norm(A)

    # The indices of mu and phi start at 1 and are thus shifted by 1 compared to the reference.
    mu_ = Dict{Int, Int}()
    mu_[0] = mu = -m
    phi_ = Dict{Int, Int}()
    phi_[0] = phi = -p
    l = 1
    n_ = [1]
    V_ = AutosizeMatrix{T}[]
    W_ = AutosizeMatrix{T}[]
    v_ = Vector{T}[]
    w_ = Vector{T}[]
    I_v = Set{Int}()
    I_w = Set{Int}()

    fac = 10

    Delta_ = Matrix{T}[]

    # The column index is greater by m than the one in the reference.
    t_ = AutosizeMatrix{T}()
    # The column index is greater by mp than the one in the reference.
    # u_ is called \tilde t in the reference.
    u_ = AutosizeMatrix{T}()
    
    push!(V_, zeros(T, N, 0))
    push!(W_, zeros(T, N, 0))

    v = T[]
    w = T[]

    n_max = 0
    exhausted = false

    for n in Iterators.countfrom(1)
        if n == stop_at
            #= At n = stop_at, it is assumed that the method should have terminated by now 
            and an error occured. This is usually caused by rounding errors due to 
            unsufficient precision. =#
            error("n = $stop_at, there seems to be a problem with deflation.")
        end
        if !exhausted
            n_max = n
        end
        # 1. Build the unnormalized right Lanczos vector v
        while true
            mu += 1
            if mu == n
                break
            end
            if mu <= 0
                v = R[:, mu+m]
            else
                v = A*v_[mu]
            end

            l_v = 1
            if mu > 0
                # l_ is called l(mu) in the reference.
                l_ = findlast(n_ .<= mu)
                if haskey(phi_, n_[l_])
                    if phi_[n_[l_]] > 0
                        l_v = findlast(n_ .<= phi_[n_[l_]])
                    end
                else
                    # If w has already been exhausted due to deflation, phi_ contains no index n_[l_].
                    l_v = findlast(n_ .<= phi_[maximum(keys(phi_))]+1)
                end
            end

            I = vcat(intersect(0:l_v-1, I_v), l_v:l-1)
            for k in I
                t_[n_[k]:n_[k+1]-1, mu+m] = Delta_[k] \ (transpose(W_[k]) * v)
                v -= V_[k] * t_[n_[k]:n_[k+1]-1, mu+m]
            end

            # The second condition means: only if L is not exhausted
            if n_[l] <= n-1 && phi+1 != n
                for i = n_[l]:n-1
                    tau = v_[i]'*v / norm2(v_[i])
                    v -= v_[i]*tau
                    t_[i, mu+m] += tau
                end
            end
            if norm(v) > dtol
                break
            else
                # deflation
                #= Note that deflation often happens directly the algorithm terminates. 
                This is part of the algorithm and no real deflation. To avoid confusion, 
                this following print statement is commented out. =#
                # println("Deflation in v at step $n (starting from 1)")
                if mu > 0 && norm(v) > 0
                    push!(I_w, l_)
                end
            end
        end
        # 2. Build the unnormalized left Lanczos vector w
        while true
            phi += 1
            if phi == n
                break
            end
            if phi <= 0
                w = L[:, phi+p]
            else
                w = transpose(A)*w_[phi]
            end

            l_w = 1
            if phi > 0
                # l_ is called l(phi) in the reference.
                l_ = findlast(n_ .<= phi)
                if haskey(mu_, n_[l_])
                    if mu_[n_[l_]] > 0
                        l_w = findlast(n_ .<= mu_[n_[l_]])
                    end
                else
                    # If v has already been exhausted due to deflation, mu_ contains no index n_[l_].
                    l_w = findlast(n_ .<= mu_[maximum(keys(mu_))]+1)
                end
            end

            I = vcat(intersect(0:l_w-1, I_w), l_w:l-1)
            for k in I
                u_[n_[k]:n_[k+1]-1, phi+p] = transpose(Delta_[k]) \ (transpose(V_[k]) * w)
                w -= W_[k] * u_[n_[k]:n_[k+1]-1, phi+p]
            end

            if n_[l] <= n-1 && mu != n
                # The second condition means: only if R is not exhausted
                for i = n_[l]:n-1
                    tau = w_[i]'*w / norm2(w_[i])
                    w -= w_[i]*tau
                    u_[i, phi+p] += tau
                end
            end
            if norm(w) > dtol
                break
            else
                # deflation
                #= Note that deflation often happens directly the algorithm terminates. 
                This is part of the algorithm and no real deflation. To avoid confusion, 
                this following print statement is commented out. =#
                # println("Deflation in w at step $n (starting from 1)")
                if phi > 0 && norm(w) > 0
                    push!(I_v, l_)
                end
            end
        end
        if mu == n_max && phi == n_max
            #= Unlike the original algorithm, the results are only returned here when both
            sides are exhausted.
            n_max equals n until one side is exhausted; then it is fixed to its current value. =#
            return Matrix(t_), Matrix(u_), v_, w_, Delta_
        end

        #= 3. Normalize v and w to obtain the n-th pair of Lanczos vectors v_n and w_n,
        and add them to the current clusters.) =#
        if phi != n
            u_[n, phi+p] = norm(w)
            push!(w_, w / u_[n, phi+p])
            W_[l] = [W_[l] w_[n]]
        end
        if mu != n
            t_[n, mu+m] = norm(v)
            push!(v_, v / t_[n, mu+m])
            V_[l] = [V_[l] v_[n]]
        end

        # 4. Record the n-th history indices.
        if phi != n
            phi_[n] = phi
        end
        if mu != n
            mu_[n] = mu
        end

        if mu == n || phi == n
            #= If any side is exhausted, only new candidate vectors for the other side are
            generated and all other steps are skipped. =#
            exhausted = true
            continue
        end

        # 5. Compute Delta_l and check for end of look-ahead clusters.
        push!(Delta_, transpose(W_[l])*V_[l])

        # if cond(Float64.(Delta[ell])) < 1/dtol
        if max(maximum(abs.(t_[:, mu+m])), maximum(abs.(u_[:, phi+p]))) <= fac*norm_A
            l += 1
            push!(n_, n+1)
            push!(V_, zeros(T, N, 0))
            push!(W_, zeros(T, N, 0))
        else
            #= In the case of look-ahead, Delta_ has to be replaced in the next step 
            of the algorithm, which is why it is deleted here. =#
            println("Look-ahead at step step $n (starting from 1)")
            pop!(Delta_)
        end
    end
end
