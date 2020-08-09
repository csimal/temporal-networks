using LightGraphs
using NetworkEpidemics
using SparseArrays
using DifferentialEquations
using LinearAlgebra


function individual_based_continuous(g::SnapshotList, β, γ, x0; tmax=100.0, saveat=[])
    N = nv(g.snapshots[1])
    S₀ = float(2.0.-x0)
    I₀ = 1.0 .- S₀
    u₀ = [S₀; I₀]
    ds = Vector{Float64}(undef, N)
    c = Vector{Float64}(undef, N)
    f! = function(du,u,p,t)
        A::SparseMatrixCSC{Int,Int} = adjacency_matrix(snapshot(g,t))
        mul!(c, A, u[(N+1):(2*N)]) # c = A*u[N+1:2*N]
        @. ds = β * u[1:N] * c
        @. du[1:N] = -ds
        @. du[N+1:2*N] = ds - γ*u[(N+1):(2*N)]
    end
    prob = ODEProblem(f!, u₀, (0.0,tmax))
    #sol = solve(prob, TRBDF2(autodiff=false), saveat=saveat, tstops=g.timestamps)
    sol = solve(prob, Tsit5(), saveat=saveat, tstops=g.timestamps)
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
        I = @view u[(L+1):(2*L)]
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
    #sol = solve(prob, TRBDF2(autodiff=false), saveat=saveat)
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
        I = @view u[(L+1):(2*L)]
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
    #jac! = function(J,u,p,t)
    #    A::SparseMatrixCSC{Int,Int} = adjacency_matrix(snapshot(g,t))
    #    θ = @view u[1:L]
    #    I = @view u[(L+1):(2*L)]
    #    R = @view u[(2*L+1):(2*L+N)]
    #    S .= z
    #    @. J = 0.0
    #    for i in 1:L
    #        (k,l) = links[i]
    #        S[l] *= θ[i]
    #        J[i,L+i] = -β*A[k,l]
    #        J[L+i,L+i] = -(γ+β*A[k,l])
    #    end
    #    for i in 1:L
    #        (k,l) = links[i]
    #        Skl = S[k]/θ[idx[l,k]]
    #        tmp = 0.0
    #        for j in neighbors(ag, k)
    #            J[L+i,L+idx[j,k]] = β*Skl*A[j,k]/θ[idx[j,k]]
    #            J[(2*L+k),idx[j,k]] = -γ*Skl/θ[idx[j,k]]
    #            tmp += A[j,k]*I[idx[j,k]]/θ[idx[j,k]]
    #        end
    #        for j in neighbors(ag, k)
    #            J[L+i,idx[j,k]] = β*(Skl/θ[idx[j,k]])*(tmp - A[j,k]*I[idx[j,k]]/θ[idx[j,k]])
    #        end
    #    end
    #    @. J[(2*L+1):(2*L+N)] = -γ
    #end
    #f = ODEFunction(f!, jac=jac!)
    prob = ODEProblem(f!, u₀, (0.0, tmax))
    #sol = solve(prob, TRBDF2(autodiff=false), saveat=saveat, tstops=g.timestamps)
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
