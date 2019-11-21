using LightGraphs
using LightGraphs.SimpleGraphs
using SparseArrays
using Random

include("TemporalNetwork.jl")

@enum EpidemicState susceptible infected recovered

struct InfectionEvent
    source
    target
    time::Real
end

struct RecoveryEvent
    node
    time::Real
end

"""
    epidemic_step_sir(g, α, β, state, [log_events::Bool=false, t=0])

Compute a single step of SIR dynamics on the network g, from the epidemic
state `state`. Returns the new state and a vector of epidemic events
"""
function epidemic_step_sir(g::AbstractSimpleGraph, α::Real, β::Real,
                           state::Vector{EpidemicState}, t=0)
    new_state = copy(state)
    events = Vector()

    for v in shuffle(vertices(g)) # go through nodes in a random order
        if new_state[v] == susceptible
            for u in [i for i in neighbors(g,v) if state[i]==infected]
                if rand() < α
                    new_state[v] = infected
                    push!(events,InfectionEvent(u,v,t))
                    break
                end
            end
        elseif new_state[v] == infected && rand() < β
            new_state[v] = recovered
            push!(events,RecoveryEvent(v,t))
        end
    end
    return new_state, events
end


"""
    simulation_sir(g, α, β, state)

Simulates and SIR epidemic process on the static network g from the initial state `state` until no node is infected. Returns the list of epidemic events
"""
function simulation_sir(g::AbstractSimpleGraph, α::Real, β::Real, state::Vector{EpidemicState})
    t = 0
    state_log = Vector()
    epidemic_log = Vector()
    while any(i->i==infected, state)
        (state,events) = epidemic_step_sir(g, α, β, state, t)
        push!(state_log, state)
        append!(epidemic_log, events)
        t += 1
    end
    return state_log, epidemic_log
end

function simulation_sir(g::TemporalNetwork, α::Real, β::Real, state::Vector{EpidemicState})
    n = length(g.snapshots)
    t = 0
    state_log = Vector()
    epidemic_log = Vector()
    while any(i->i==infected, state) && t+1 < n
        (state,events) = epidemic_step_sir(g.snapshots[t+1], α, β, state, t)
        push!(state_log, state)
        append!(epidemic_log, events)
        t += 1
    end
    return state_log, epidemic_log
end

function montecarlo_sir(g, α::Real, β::Real, state::Vector{EpidemicState}, max::Integer)
    n = length(state)
    # probabilities of each being in a given state at time t
    s = Vector{Vector{Integer}}()
    i = Vector{Vector{Integer}}()
    r = Vector{Vector{Integer}}()
    # probabilities of epidemic events
    infection = Vector{SparseMatrixCSC}()
    recovery = Vector{Vector{Integer}}()

    for k in 1:max
        states, events = simulation_sir(g, α, β, state)
        t_end = length(states)
        while t_end > length(s) # pad arrays if necessary
            push!(s, zeros(Int64,n))
            push!(i, zeros(Int64,n))
            push!(r, zeros(Int64,n))
            push!(infection, spzeros(Int64,n,n))
            push!(recovery, zeros(Int64,n))
        end
        for t in 1:t_end, k in 1:n
            if states[t][k] == susceptible
                s[t][k] += 1
            elseif states[t][k] == infected
                i[t][k] += 1
            else
                r[t][k] += 1
            end
        end
        for e in events
            if typeof(e) == InfectionEvent
                infection[e.time+1][e.source,e.target] += 1
            else
                recovery[e.time+1][e.node] += 1
            end
        end
    end
    return s/max, i/max, r/max, infection/max, recovery/max
end

function epidemic_step_ib(g::AbstractSimpleGraph, α::Real, β::Real, state)
    (s,i,r) = state # probability vectors for each possible epidemic state
    n = length(s)
    s_new = zeros(n)
    i_new = zeros(n)
    r_new = zeros(n)
    A = adjacency_matrix(g)

    for v in vertices(g)
        s_new[v] = s[v]*prod(u->1-α*i[u], filter(i->A[i,v]!=0.0, 1:n))
        i_new[v] = (1-β)*i[v] + s[v] - s_new[v]
        r_new[v] = 1 - s_new[v] - i_new[v]
    end
    return (s_new, i_new, r_new)
end

function epidemic_step_cb(g::AbstractSimpleGraph, α::Real, β::Real, state)
    (θ,s,i,r,z) = state # NB. the state is described here by edge quantities contained in square matrices
    n = size(θ,1)
    θ_new = spzeros(n,n) # empty sparse matrices
    s_new = spzeros(n,n)
    i_new = spzeros(n,n)
    r_new = spzeros(n,n)
    A = adjacency_matrix(g)

    θ_new = θ - α*A.*i # element wise product
    v = vertices(g)
    for k in v, l in v
        s_new[k,l] = z[k]*prod(setdiff(inneighbors(g,k),l))
    end
    i_new = (1-β)*(I-α*A).*i + s - s_new
    r_new = r + β*i
    return θ_new, s_new, i_new, r_new, z # must pass z again for easy iteration
end
