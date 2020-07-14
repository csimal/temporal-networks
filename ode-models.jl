using LightGraphs
using NetworkEpidemics
using SparseArrays
using DifferentialEquations


function individual_based_continuous(g::SnapshotList, β, γ, x0; tmax=100.0, saveat=[])
    N = nv(g.snapshots[1])
    S₀ = float(2.0.-x0)
    I₀ = 1.0 .- S₀
    u₀ = [S₀; I₀]
    ds = Vector{Float64}(undef, N)
    f! = function(du,u,p,t)
        A::SparseMatrixCSC{Int,Int} = adjacency_matrix(snapshot(g,t))
        @. ds = β * u[1:N] * (A*u[N+1:2*N])
        @. du[1:N] = -ds
        @. du[N+1:2*N] = ds - γ*u[N+1:2*N]
    end
    prob = ODEProblem(f!, u₀, (0.0,tmax))
    sol = solve(prob, ABDF2(), saveat=saveat, tstops=g.timestamps)
    S = Array{Float64,2}(undef, length(sol.t), N)
    I = Array{Float64,2}(undef, length(sol.t), N)
    R = Array{Float64,2}(undef, length(sol.t), N)
    for t in 1:length(sol.t)
        S[t,:] .= sol.u[t][1:N]
        I[t,:] .= sol.u[t][N+1:2*N]
        R[t,:] .= 1.0 .- S[t,:] .- I[t,:]
    end
    return sol.t, [S, I, R]
end

function contact_based_continuous(cp::ContactProcess{SIR}, x0, tmax; saveat=[])
    g = SimpleDiGraph(cp.g)
    N = nv(g)
    L = ne(g)
    links = map(Tuple,edges(g))
    β = cp.dynamics.β
    γ = cp.dynamics.δ
    θ₀ = ones(L)
    I₀ = zeros(L)
    R₀ = zeros(N)
    z = float(2.0.-x0)
    idx = spzeros(Int, N, N) # nodes to link index
    for i in 1:L
        (k,l) = links[i]
        idx[k,l] = i
        θ₀[i] = 1.0
        I₀[i] = 1.0 - z[k]
    end
    u₀ = [θ₀; I₀; R₀]
    S = Vector{Float64}(undef, N)
    f! = function(du,u,p,t)
        θ = @view u[1:L]
        I = @view u[(L+1):2*L]
        R = @view u[(2*L+1):(2*L+N)]
        S .= z
        for i in 1:L
            (k,l) = links[i]
            S[l] *= θ[i]
        end
        for i in 1:L
            (k,l) = links[i]
            du[i] = -β * I[i]
            du[L+i] = -(γ+β)*I[i] + β*(S[k]/(θ[idx[l,k]])) * sum(vcat([0],[I[idx[j,k]]/θ[idx[j,k]] for j in neighbors(g, k) if j != l ]))
            du[(2*L+1):(2*L+N)] .= γ.*(1.0 .- S .- R)
        end
    end
    prob = ODEProblem(f!, u₀, (0.0, tmax))
    sol = solve(prob, Tsit5(), saveat=saveat)
    S = Array{Float64,2}(undef, length(sol.t), N)
    I = Array{Float64,2}(undef, length(sol.t), N)
    R = Array{Float64,2}(undef, length(sol.t), N)
    for t in 1:length(sol.t)
        R[t,:] .= sol.u[t][(2*L+1):(2*L+N)]
        S[t,:] .= z
        for i in 1:L
            (k,l) = links[i]
            S[t,l] *= sol.u[t][i]
        end
        I[t,:] .= 1.0 .- S[t,:] .- R[t,:]
    end
    return sol.t, [S, I, R]
end

function contact_based_continuous(g::SnapshotList, β::Real, γ::Real, x0; tmax=100.0, saveat=[])
    ag = SimpleDiGraph(aggregate_network(g))
    N = nv(ag)
    L = ne(ag)
    links = map(Tuple,edges(ag))
    θ₀ = ones(L)
    I₀ = zeros(L)
    R₀ = zeros(N)
    z = float(2.0.-x0)
    idx = spzeros(Int, N, N) # nodes to link index
    for i in 1:L
        (k,l) = links[i]
        idx[k,l] = i
        θ₀[i] = 1.0
        I₀[i] = 1.0 - z[k]
    end
    u₀ = [θ₀; I₀; R₀]
    S = Vector{Float64}(undef, N)
    f! = function(du,u,p,t)
        A::SparseMatrixCSC{Int,Int} = adjacency_matrix(snapshot(g,t))
        θ = @view u[1:L]
        I = @view u[(L+1):2*L]
        R = @view u[(2*L+1):(2*L+N)]
        S .= z
        for i in 1:L
            (k,l) = links[i]
            S[l] *= θ[i]
        end
        for i in 1:L
            (k,l) = links[i]
            du[i] = -β * A[k,l] * I[i]
            du[L+i] = -(γ+β*A[k,l])*I[i] + β*(S[k]/(θ[idx[l,k]])) * sum(vcat([0],[I[idx[j,k]]/θ[idx[j,k]] for j in neighbors(ag, k) if j != l && A[j,k] == 1 ]))
            @. du[(2*L+1):(2*L+N)] = γ*(1.0 - S - R)
        end
    end
    prob = ODEProblem(f!, u₀, (0.0, tmax))
    sol = solve(prob, Tsit5(), saveat=saveat, tstops=g.timestamps)
    S = Array{Float64,2}(undef, length(sol.t), N)
    I = Array{Float64,2}(undef, length(sol.t), N)
    R = Array{Float64,2}(undef, length(sol.t), N)
    for t in 1:length(sol.t)
        R[t,:] .= sol.u[t][(2*L+1):(2*L+N)]
        S[t,:] .= z
        for i in 1:L
            (k,l) = links[i]
            S[t,l] *= sol.u[t][i]
        end
        I[t,:] .= 1.0 .- S[t,:] .- R[t,:]
    end
    return sol.t, [S, I, R]
end

function path_based_continuous(cp::ContactProcess{SIR}, x0, tmax)
    g = SimpleDiGraph(cp.g)
    N = nv(g)
    β = cp.dynamics.β
    γ = cp.dynamics.γ
    f! = function(du,u,p,t)

    end
end
