using LightGraphs
using LightGraphs.SimpleGraphs

include("utils.jl")

struct Contact
     time::Float64
     duration::Float64
     edge::Edge
end

abstract type AbstractTemporalNetwork end

struct ContactList <: AbstractTemporalNetwork
    contacts::Vector{Contact}
    nv::Int
end

struct SnapshotList <: AbstractTemporalNetwork
    snapshots::Vector{SimpleGraph}
    timestamps::Vector{Float64}
end

function ContactList(xs::Vector{Contact})
    max = 0
    for x in xs
        max = maximum(max, src(x.edge), dst(x.edge))
    end
    return ContactList(sort(x), max)
end

function neighborhoods(g::ContactList)
    N = g.nv
    out = [Dict{Int,Vector{Tuple{Float64,Float64}}}() for u in 1:N]
    for c in g.contacts
        u = src(c.edge)
        v = dst(c.edge)
        if !haskey(out[u],v)
            out[u][v] = [(c.time,c.duration)]
        else
            push!(out[u][v], (c.time,c.duration))
        end
        if !haskey(out[v], u)
            out[v][u] = [(c.time,c.duration)]
        else
            push!(out[v][u], (c.time,c.duration))
        end
    end
    return out
end

function neighborhoods(g::SnapshotList)
    N = nv(g.snapshots[1])
    ts = g.timestamps
    out  = [Dict{Int,Vector{Tuple{Float64,Float64}}}() for u in 1:N]
    for k in 1:length(ts)-1, e in edges(g.snapshots[k])
        u = src(e)
        v = dst(e)
        if !haskey(out[u], v)
            out[u][v] = [(ts[k], ts[k+1])]
        else
            push!(out[u][v], (ts[k], ts[k+1]))
        end
        if !haskey(out[v], u)
            out[v][u] = [(ts[k], ts[k+1])]
        else
            push!(out[v][u], (ts[k], ts[k+1]))
        end
    end
    return out
end

function snapshot(g::ContactList, tlb::Real, tub::Real)
    edges = [e for (t,dt,e) in g.contacts if t >= tlb && t+dt < tub]
    return SimpleGraph(edges)
end

function SnapshotList(g::ContactList)
    times = PriorityQueue{Float64,Float64}()
    for c in g.contacts
        enqueue!(times, c.time, c.time)
        enqueue!(times, c.time+c.duration, c.time+c.duration)
    end
    timestamps = Vector{Float64}(undef, length(times))
    for i in 1:length(timestamps)
        timestamps[i] = dequeue!(times)
    end
    snapshots = [SimpleGraph(g.nv) for i in 1:T]
    for c in g.contacts
        i = 1
        while timestamps[i] < c.time
            i += 1
        end
        while timestamps[i] < c.time + c.duration
            add_edge!(snapshots[i], c.edge)
            i += 1
        end
    end
    return SnapshotList(snapshots, timestamps)
end

SnapshotList(g::SimpleGraph, tmax::Float64) = SnapshotList([g], [0.0,tmax])

function ContactList(g::SnapshotList)
    ag = aggregate_network(g)
    contacts = []
    for e in edges(ag)
        active = false
        time = 0.0
        for t in 1:length(g.snapshots)
            if !active && has_edge(g.snapshots[t], e)
                active = true
                time = g.timestamps[t]
            elseif active && ( (!has_edge(g.snapshots[t], e)) || t==length(g.snapshots))
                active = false
                duration = 0.0
                if !has_edge(g.snapshots[t], e)
                    duration = g.timestamps[t] - time
                else
                    duration = g.timestamps[t+1] - time
                end
                push!(contacts, Contact(time, duration, e))
            end
        end
    end
    sort!(contacts, by = x -> (x.time,x.duration)) # sort in lexicographic order of time-duration
    return ContactList(contacts, nv(ag))
end

snapshots(g::SnapshotList) = g.snapshots

function snapshot(g::SnapshotList, t::Real)
    if t < g.timestamps[1] || t >= g.timestamps[length(g.timestamps)]
        return SimpleGraph(nv(g.snapshots[1]))
    end
    k = bissection_search(g.timestamps, float(t))
    return g.snapshots[k]
end

function aggregate_network(g::ContactList)
    return SimpleGraphFromIterator(map(c->c.edge, g.contacts))
end

function aggregate_network(g::SnapshotList)
    h = SimpleGraph(g.snapshots[1])
    for t in 2:length(g.snapshots), e in edges(g.snapshots[t])
        add_edge!(h, e)
    end
    return h
end
