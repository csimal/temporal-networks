using LightGraphs
using LightGraphs.SimpleGraphs
using Random

include("temporal-network.jl")

function activity_driven(N::Integer, T::Integer, Δt::Real, x::Vector{Float64}, k::Int, m::Integer)
    snapshots = [SimpleGraph(N) for t in 1:T]
    a = x*k/(mean(x)*N)
    for t in 1:T, i in 1:N
        if rand() <= a[i]*Δt
            neighbors = shuffle!(deleteat!(collect(1:N), i))[1:m]
            for j in neighbors
                add_edge!(snapshots[t], i,j)
            end
        end
    end
    return SnapshotList(snapshots, collect(Δt*(0:T)))
end

function random_sequence(N::Integer, T::Integer, Δt::Real, f = (n->erdos_renyi(n,0.5)))
    snapshots = Vector{SimpleGraph}(undef, T)
    for t in 1:T
        snapshots[t] = f(N)
    end
    return SnapshotList(snapshots, collect(Δt*(0:T)))
end

function random_contacts(g::SimpleGraph, T::Integer, Δt::Real, p=0.5)
    snapshots = [SimpleGraph(nv(g)) for t in 1:T]
    for t in 1:T, e in edges(g)
        if rand() < p
            add_edge!(snapshots[t], e)
        end
    end
    return SnapshotList(snapshots, collect(Δt*(0:T)))
end
