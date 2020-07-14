using LightGraphs
using NetworkEpidemics
using DifferentialEquations

function pair_approximation(cp::ContactProcess{SIR}, x0; tmax=100.0, saveat=[])
    g = cp.g
    N = nv(g)
    L = ne(g)
    links = map(Tuple, edges(g))
    β = cp.dynamics.β
    γ = cp.dynamics.δ
    S₀ = float(2.0.-x0)
    I₀ = 1.0.-S₀
    SS₀ = Vector{Float64}(undef, L)
    SI₀ = Vector{Float64}(undef, 2*L)
    ds = Vector{Float64}(undef, N)
    for i in 1:L
        (k,l) = links[i]
        SS₀[i] = S₀[k]*S₀[l]
        SI₀[i] = S₀[k]*I₀[l]
        SI₀[L+i] = I₀[k]*S₀[l]
    end
    u₀ = [S₀; I₀; SS₀; SI₀]
    f! = function(du,u,p,t)
        S = @view u[1:N]
        I = @view u[(N+1):2*N]
        SS = @view u[(2*N+1):(2*N+L)]
        SI = @view u[(2*N+L+1):(2*N+3*L)]
        ds .= 0.0
        ds1::Float64 = 0.0
        ds2::Float64 = 0.0
        for i in 1:L
            (k,l) = links[i]
            ds[k] += SI[i]
            ds[l] += SI[L+i]
        end
        @. du[1:N] = -β*ds
        @. du[(N+1):2*N] = β*ds - γ*I
        for i in 1:L
            (k,l) = links[i]
            if S[k] != 0.0
                ds1 = (β/S[k])*(ds[k]-SI[i])
            else
                ds1 = 0.0
            end
            if S[l] != 0.0
                ds2 = (β/S[l])*(ds[l]-SI[L+i])
            else
                ds2 = 0.0
            end
            du[2*N+i] = -SS[i]*ds1 - SS[i]*ds2
            du[2*N+L+i] = SS[i]*ds2 - SI[i]*ds1 - (β+γ)*SI[i]
            du[2*N+2*L+i] = SS[i]*ds1 - SI[L+i]*ds2 - (β+γ)*SI[L+i]
        end
    end
    prob = ODEProblem(f!, u₀, (0.0, tmax))
    sol = solve(prob, Tsit5(), saveat=saveat)
    S = Array{Float64,2}(undef, length(sol.t), N)
    I = Array{Float64,2}(undef, length(sol.t), N)
    R = Array{Float64,2}(undef, length(sol.t), N)
    for t in 1:length(sol.t)
        S[t,:] .= sol.u[t][1:N]
        I[t,:] .= sol.u[t][(N+1):2*N]
        R[t,:] .= 1.0 .- S[t,:] .- I[t,:]
    end
    return sol.t, [S, I, R]
end

function pair_approximation(g::SnapshotList, β, γ, x0; tmax=100.0, saveat=[])
    ag = aggregate_network(g)
    N = nv(ag)
    L = ne(ag)
    links = map(Tuple, edges(ag))
    S₀ = float(2.0.-x0)
    I₀ = 1.0.-S₀
    SS₀ = Vector{Float64}(undef, L)
    SI₀ = Vector{Float64}(undef, 2*L)
    for i in 1:L
        (k,l) = links[i]
        SS₀[i] = S₀[k]*S₀[l]
        SI₀[i] = S₀[k]*I₀[l]
        SI₀[L+i] = I₀[k]*S₀[l]
    end
    u₀ = [S₀; I₀; SS₀; SI₀]
    ds = Vector{Float64}(undef, N)
    f! = function(du,u,p,t)
        A::SparseMatrixCSC{Int64,Int64} = adjacency_matrix(snapshot(g, t))
        S = @view u[1:N]
        I = @view u[(N+1):2*N]
        SS = @view u[(2*N+1):(2*N+L)]
        SI = @view u[(2*N+L+1):(2*N+3*L)]
        ds .= 0.0
        ds1::Float64 = 0.0
        ds2::Float64 = 0.0
        for i in 1:L
            (k,l) = links[i]
            ds[k] += A[l,k]*SI[i]
            ds[l] += A[k,l]*SI[L+i]
        end
        @. du[1:N] = -β*ds
        @. du[(N+1):2*N] = β*ds - γ*I
        for i in 1:L
            (k,l) = links[i]
            if S[k] != 0.0
                ds1 = (β/S[k])*(ds[k]-A[l,k]*SI[i])
            else
                ds1 = 0.0
            end
            if S[l] != 0.0
                ds2 = (β/S[l])*(ds[l]-A[k,l]*SI[L+i])
            else
                ds2 = 0.0
            end
            du[2*N+i] = -SS[i]*ds1 - SS[i]*ds2
            du[2*N+L+i] = SS[i]*ds2 - SI[i]*ds1 - (A[l,k]*β+γ)*SI[i]
            du[2*N+2*L+i] = SS[i]*ds1 - SI[L+i]*ds2 - (A[k,l]*β+γ)*SI[L+i]
        end
    end
    prob = ODEProblem(f!, u₀, (0.0, tmax))
    sol = solve(prob, Tsit5(), saveat=saveat, tstops=g.timestamps)
    S = Array{Float64,2}(undef, length(sol.t), N)
    I = Array{Float64,2}(undef, length(sol.t), N)
    R = Array{Float64,2}(undef, length(sol.t), N)
    for t in 1:length(sol.t)
        S[t,:] .= sol.u[t][1:N]
        I[t,:] .= sol.u[t][(N+1):2*N]
        R[t,:] .= 1.0 .- S[t,:] .- I[t,:]
    end
    return sol.t, [S, I, R]
end

function pair_approximation_fun(g::SnapshotList, β, γ, x0)
    ag = aggregate_network(g)
    N = nv(ag)
    L = ne(ag)
    links = map(Tuple, edges(ag))
    S₀ = float(2.0.-x0)
    I₀ = 1.0.-S₀
    SS₀ = Vector{Float64}(undef, L)
    SI₀ = Vector{Float64}(undef, 2*L)
    for i in 1:L
        (k,l) = links[i]
        SS₀[i] = S₀[k]*S₀[l]
        SI₀[i] = S₀[k]*I₀[l]
        SI₀[L+i] = I₀[k]*S₀[l]
    end
    u₀ = [S₀; I₀; SS₀; SI₀]
    ds = Vector{Float64}(undef, N)
    f! = function(du,u,p,t)
        A::SparseMatrixCSC{Int64,Int64} = adjacency_matrix(snapshot(g, t))
        S = @view u[1:N]
        I = @view u[(N+1):2*N]
        SS = @view u[(2*N+1):(2*N+L)]
        SI = @view u[(2*N+L+1):(2*N+3*L)]
        ds .= 0.0
        ds1::Float64 = 0.0
        ds2::Float64 = 0.0
        for i in 1:L
            (k,l) = links[i]
            ds[k] += A[l,k]*SI[i]
            ds[l] += A[k,l]*SI[L+i]
        end
        @. du[1:N] = -β*ds
        @. du[(N+1):2*N] = β*ds - γ*I
        for i in 1:L
            (k,l) = links[i]
            if S[k] != 0.0
                ds1 = (β/S[k])*(ds[k]-A[l,k]*SI[i])
            else
                ds1 = 0.0
            end
            if S[l] != 0.0
                ds2 = (β/S[l])*(ds[l]-A[k,l]*SI[L+i])
            else
                ds2 = 0.0
            end
            du[2*N+i] = -SS[i]*ds1 - SS[i]*ds2
            du[2*N+L+i] = SS[i]*ds2 - SI[i]*ds1 - (A[l,k]*β+γ)*SI[i]
            du[2*N+2*L+i] = SS[i]*ds1 - SI[L+i]*ds2 - (A[k,l]*β+γ)*SI[L+i]
        end
    end
    return f!, u₀
end
