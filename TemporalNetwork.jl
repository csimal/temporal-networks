using LightGraphs
using LightGraphs.SimpleGraphs

struct Contact
     time::Integer
     edge::Edge
end

struct TemporalEdgeList
    edges::Vector{Contact}
end

struct TemporalNetwork
    snapshots::Vector{AbstractSimpleGraph}
    timestamps::Vector{Real}
end

function snapshot(network::TemporalEdgeList, time::Integer)
    edges = [e for (t,e) in network.edges if t==time]
    return SimpleGraph(edges)
end

function snapshots(network::TemporalEdgeList)
    size = maximum(first, network.edges)
    networks = Vector{SimpleGraph}
    for t in 1:size
        networks[t] = snapshot(network, t)
    end
    return networks
end

function aggregate_network(tnet::TemporalEdgeList)
    return SimpleGraphFromIterator(map(c->c.edge, tnet.edges))
end
