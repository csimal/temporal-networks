using DataStructures
using LightGraphs
using ProgressMeter
using Statistics

include("temporal-network.jl")
include("utils.jl")

abstract type EpidemicEvent end

struct InfectionEvent <: EpidemicEvent
    node::Int
end

struct RecoveryEvent <: EpidemicEvent
    node::Int
end


function event_based_dtsir(g::SnapshotList, β::Real, γ::Real, x0)
    N = nv(g.snapshots[1])
    Δt = g.timestamps[2]
    neighbors = neighborhoods(g)
    events = PriorityQueue{EpidemicEvent,Float64}()
    ts = Vector{Float64}(undef, 2*N)
    S = Vector{Int}(undef, 2*N)
    I = Vector{Int}(undef, 2*N)
    R = Vector{Int}(undef, 2*N)
    k = 1
    ts[k] = 0.0
    S[k] = N
    I[k] = 0
    R[k] = 0
    state = copy(x0)
    for u in 1:N
        if x0[u] == 2
            events[InfectionEvent(u)] = 0.0
        end
    end
    while !isempty(events)
        (e,t)  = dequeue_pair!(events)
        if t != ts[k]
            k += 1
            ts[k] = t
            S[k] = S[k-1]
            I[k] = I[k-1]
            R[k] = R[k-1]
        end
        if typeof(e) == InfectionEvent
            S[k] -= 1
            I[k] += 1
            u = e.node
            state[u] = 2
            if γ != 0.0
                τᵣ = Δt*(1+floor(log(1.0-rand())/log(1.0-Δt*γ))) # recovery time of u
                events[RecoveryEvent(u)] = t+τᵣ
            else
                τᵣ = Inf
            end
            for v in filter(x->state[x]==1, keys(neighbors[u]))
                contacts = map(first, neighbors[u][v])
                imin = bissection_search(contacts, t) # latest contact before t
                if imin < length(neighbors[u][v]) && contacts[imin+1] <= t+τᵣ
                    imax = bissection_search(contacts, t+τᵣ)
                    if haskey(events, InfectionEvent(v))
                        i = bissection_search(contacts, events[InfectionEvent(v)])
                        if i <= imax
                            imax = i-1
                        end
                    end
                    iᵢ = 1 + Int64(floor(log(1.0-rand())/log(1.0-Δt*β)))
                    if iᵢ < imax - imin
                        events[InfectionEvent(v)] = contacts[imin+iᵢ]
                    end
                end
            end
        else # recovery
            I[k] -= 1
            R[k] += 1
        end
    end
    return ts[1:k], [S[1:k], I[1:k], R[1:k]]
end

function average_dtsir(g::SnapshotList, β::Real, γ::Real, x0; nsims=500, nbins=100, tmax=100.0, progressbar=false)
    ts = LinRange(0.0, tmax, nbins)
    sims = Array{Int,3}(undef, nsims, 3, nbins)
    if progressbar
        p = Progress(nsims, dt=1.0)
    end
    for n in 1:nsims
        t, states = event_based_dtsir(g, β, γ, x0)
        l = 1
        for k in 1:nbins
            while l < length(t) && t[l+1] <= ts[k]
                l += 1
            end
            sims[n,1,k] = states[1][l]
            sims[n,2,k] = states[2][l]
            sims[n,3,k] = states[3][l]
        end
        if progressbar
            ProgressMeter.next!(p; showvalues = [(:iteration,n)])
        end
    end
    μ = reshape(mean(sims, dims=1), 3, nbins)
    σ = reshape(std(sims, mean=μ, dims=1), 3, nbins)
    return ts, μ, σ
end
