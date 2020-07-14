using LightGraphs
using NetworkEpidemics
using SparseArrays


function individual_based_discrete(cp::ContactProcess, x0, tmax::Integer, Δt=1.0)
    g = cp.g
    N = nv(g)
    β = cp.dynamics.β
    γ = cp.dynamics.δ
    z = float(2.0.-x0)
    S = Array{Float64,2}(undef, tmax+1, N)
    I = Array{Float64,2}(undef, tmax+1, N)
    R = Array{Float64,2}(undef, tmax+1, N)
    for k in vertices(g)
        S[1,k] = z[k]
        I[1,k] = 1.0 - z[k]
        R[1,k] = 0.0
    end
    for t in 1:tmax
        @. S[t+1,:] = S[t,:]
        for k in 1:N, l in outneighbors(g,k)
            S[t+1,k] *= 1.0 - β*Δt*I[t,l]
        end
        @. R[t+1,:] = R[t,:] + γ*Δt*I[t,:]
        @. I[t+1,:] = 1.0 - S[t+1,:] - R[t,:]
    end
    return Δt*(0:tmax), [S, I, R]
end

function individual_based_discrete(g::SnapshotList, β::Real, γ::Real, x0, tmax::Integer, Δt=1.0)
    N = nv(g.snapshots[1])
    z = float(2.0.-x0)
    S = Array{Float64,2}(undef, tmax+1, N)
    I = Array{Float64,2}(undef, tmax+1, N)
    R = Array{Float64,2}(undef, tmax+1, N)
    for k in 1:N
        S[1,k] = z[k]
        I[1,k] = 1.0 - z[k]
        R[1,k] = 0.0
    end
    for t in 1:tmax
        S[t+1,:] .= S[t,:]
        if t < length(g.timestamps)
            for k in 1:N, l in neighbors(g.snapshots[t],k)
                S[t+1,k] *= (1.0 - β*Δt*I[t,l])
            end
        end
        @. R[t+1,:] =  R[t,:] + γ*Δt*I[t,:]
        @. I[t+1,:] = 1.0 - S[t+1,:] - R[t+1,:]
    end
    return Δt*(0:tmax), [S, I, R]
end

function contact_based_discrete(cp::ContactProcess{SIR}, x0, tmax::Integer, Δt=1.0)
    g = SimpleDiGraph(cp.g)
    N = nv(g)
    L = ne(g)
    links = map(Tuple, edges(g))
    idx = spzeros(Int,N,N)
    β = cp.dynamics.β
    γ = cp.dynamics.δ
    z = float(2.0.-x0)
    θ = Array{Float64,2}(undef, tmax+1, L)
    Iₑ = Array{Float64,2}(undef, tmax+1, L)
    Sₑ = Array{Float64,}(undef, L)
    S = Array{Float64,2}(undef, tmax+1, N)
    I = Array{Float64,2}(undef, tmax+1, N)
    R = Array{Float64,2}(undef, tmax+1, N)
    for k in vertices(g)
        S[1,k] = z[k]
        I[1,k] = 1.0-z[k]
        R[1,k] = 0.0
    end
    for e in 1:L
        (k,l) = links[e]
        idx[k,l] = e
        θ[1,e] = 1.0
        Iₑ[1,e] = I[1,k]
    end
    for t in 1:tmax
        @. θ[t+1,:] = θ[t,:] - β*Δt*Iₑ[t,:]
        @. S[t+1,:] = S[1,:]
        for e in 1:L
            (k,l) = links[e]
            S[t+1,l] *= θ[t+1,e]
        end
        for e in 1:L
            (k,l) = links[e]
            ΔS = (S[t,k]/θ[t,idx[l,k]]) - (S[t+1,k]/θ[t+1,idx[l,k]])
            Iₑ[t+1,e] = (1.0-γ*Δt)*(1.0-β*Δt)*Iₑ[t,e] + ΔS
        end
        @. R[t+1,:] = R[t,:] + Δt*γ*I[t,:]
        @. I[t+1,:] = 1.0 - S[t+1,:] - R[t+1,:]
    end
    return Δt*(0:tmax), [S, I, R]
end

function contact_based_discrete(g::SnapshotList, β::Real, γ::Real, x0, tmax::Integer, Δt=1.0)
    ag = SimpleDiGraph(aggregate_network(g))
    N = nv(ag)
    L = ne(ag)
    links = map(Tuple, edges(ag))
    idx = spzeros(Int,N,N)
    z = float(2.0.-x0)
    θ = Array{Float64,2}(undef, tmax+1, L)
    Iₑ = Array{Float64,2}(undef, tmax+1, L)
    Sₑ = Array{Float64,}(undef, L)
    S = Array{Float64,2}(undef, tmax+1, N)
    I = Array{Float64,2}(undef, tmax+1, N)
    R = Array{Float64,2}(undef, tmax+1, N)
    for k in 1:N
        S[1,k] = z[k]
        I[1,k] = 1.0-z[k]
        R[1,k] = 0.0
    end
    for e in 1:L
        (k,l) = links[e]
        idx[k,l] = e
        θ[1,e] = 1.0
        Iₑ[1,e] = I[1,k]
    end
    for t in 1:tmax
        if t < length(g.timestamps)
            A = adjacency_matrix(g.snapshots[t])
        else
            A = spzeros()
        end
        for e in 1:L
            (k,l) = links[e]
            θ[t+1,e] = θ[t,e] - β*Δt*A[k,l]*Iₑ[t,e]
        end
        @. S[t+1,:] = S[1,:]
        for e in 1:L
            (k,l) = links[e]
            S[t+1,l] *= θ[t+1,e]
        end
        for e in 1:L
            (k,l) = links[e]
            ΔS = (S[t,k]/θ[t,idx[l,k]]) - (S[t+1,k]/θ[t+1,idx[l,k]])
            Iₑ[t+1,e] = (1.0-γ*Δt)*(1.0-β*A[k,l]*Δt)*Iₑ[t,e] + ΔS
        end
        @. R[t+1,:] = R[t,:] + Δt*γ*I[t,:]
        @. I[t+1,:] = 1.0 - S[t+1,:] - R[t+1,:]
    end
    return Δt*(0:tmax), [S, I, R]
end
