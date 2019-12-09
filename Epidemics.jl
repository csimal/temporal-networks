using LightGraphs
using LightGraphs.SimpleGraphs
using SparseArrays
using LinearAlgebra
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
    epidemic_step_sir(g, α, β, state, [t=0])

Compute a single step of SIR dynamics on the network g, from the epidemic
state `state`. Returns the new state and a vector of epidemic events.
"""
function epidemic_step_sir(g::AbstractSimpleGraph, α::Real, β::Real, state::Vector{EpidemicState}, t=0)
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

Simulate an SIR epidemic process on the static network g from the initial state `state` until no node is infected. Returns the list of epidemic events.
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

function simulation_sir(g::AbstractTemporalNetwork, α::Real, β::Real, state::Vector{EpidemicState})
    snaps = snapshots(g)
    tmax = length(snaps)
    t = 0
    state_log = Vector()
    epidemic_log = Vector()
    while any(i->i==infected, state) && t+1 < tmax
        (state,events) = epidemic_step_sir(snaps[t+1], α, β, state, t)
        push!(state_log, state)
        append!(epidemic_log, events)
        t += 1
    end
    return state_log, epidemic_log
end

"""
    montecarlo_sir(g, α, β, state)

Perform `max` simulations of SIR dynamics on `g` (which can be either an AbstractSimpleGraph or a TemporalNetwork) starting from the initial state `state`. Returns a tuple `(s,i,r,infection,recovery)`.
`s`, `i` and `r` contain respectively the node-wise frequencies of being either susceptible, infected or recovered at each time step, i.e. `s[t][k]` contains the empirical probability of node `k` being susceptible at time `t`.
`infection[t][j,k]` contains the empirical probability of infection being transmitted from  `j` to `k` at time `t`.
`recovery[t][k]` contains the empirical probability of node `k` recovering at time `t`.
"""
function montecarlo_sir(g, α::Real, β::Real, state::Vector{EpidemicState}, max::Integer)
    n = length(state)
    # probabilities of each being in a given state at time t
    s = Vector{Vector{Integer}}()
    i = Vector{Vector{Integer}}()
    r = Vector{Vector{Integer}}()
    push!(s, max*float(state .== susceptible))
    push!(i, max*float(state .== infected))
    push!(r, max*float(state .== recovered))
    # probabilities of epidemic events
    infection = Vector{SparseMatrixCSC}()
    recovery = Vector{Vector{Integer}}()
    push!(infection, spzeros(Int64,n,n))
    push!(recovery, zeros(Int64,n))

    for k in 1:max
        states, events = simulation_sir(g, α, β, state)
        t_end = length(states)
        while length(s) < (t_end+1) # pad arrays if necessary
            push!(s, zeros(Int64,n))
            push!(i, zeros(Int64,n))
            push!(r, zeros(Int64,n))
            push!(infection, spzeros(Int64,n,n))
            push!(recovery, zeros(Int64,n))
        end
        for t in 1:(t_end), k in 1:n
            if states[t][k] == susceptible
                s[t+1][k] += 1
            elseif states[t][k] == infected
                i[t+1][k] += 1
            else
                r[t+1][k] += 1
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

"""
    epidemic_step_ib(g, α, β, state)

Compute a single step of the IB model of SIR dynamics on the network `g`.
`state` contains probability vectors `s`, `i` and `r`.
Returns the probability vectors for the next time step.
"""
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
    return s_new, i_new, r_new
end

"""
    simulation_ib(g::AbstractSimpleGraph, α::Real, β::Real, state, tmax::Integer)

Compute probabilities of the SIR dynamics on the network `g` using the IB model up to time `tmax`. `state` contains initial probability vectors `s`, `i`, `r`. Returns matrices `s_ib`, `i_ib`, `r_ib` whose rows are the successive probability vectors.
"""
function simulation_ib(g::AbstractSimpleGraph, α::Real, β::Real, state, tmax::Integer)
    n = length(state[1])
    s, i, r = state
    s_ib = zeros(tmax+1, n)
    i_ib = zeros(tmax+1, n)
    r_ib = zeros(tmax+1, n)
    s_ib[1,:] = s
    i_ib[1,:] = i
    r_ib[1,:] = r
    for j in 2:(tmax+1)
        state = epidemic_step_ib(g, α, β, state)
        s_ib[j,:] = state[1]
        i_ib[j,:] = state[2]
        r_ib[j,:] = state[3]
    end
    return s_ib, i_ib, r_ib
end

function simulation_ib(g::AbstractTemporalNetwork, α::Real, β::Real, state, tmax::Integer)
    n = length(state[1])
    snaps = snapshots(g)
    tmax = max(tmax, length(snaps))
    s, i, r = state
    s_ib = zeros(tmax+1, n)
    i_ib = zeros(tmax+1, n)
    r_ib = zeros(tmax+1, n)
    s_ib[1,:] = s
    i_ib[1,:] = i
    r_ib[1,:] = r
    for j in 2:(tmax+1)
        state = epidemic_step_ib(g[j-1], α, β, state)
        s_ib[j,:] = state[1]
        i_ib[j,:] = state[2]
        r_ib[j,:] = state[3]
    end
    return s_ib, i_ib, r_ib
end

function epidemic_step_cb(g::AbstractSimpleGraph, α::Real, β::Real, state)
    (θ,s,i,z) = state # NB. the state is described here by edge quantities contained in square matrices
    n = size(θ,1)
    θ_new = spzeros(n,n) # empty sparse matrices
    s_new = spzeros(n,n)
    i_new = spzeros(n,n)
    A = adjacency_matrix(g)

    θ_new = θ - α*A.*i # element wise product
    v = vertices(g)
    for k in v, l in v
        s_new[k,l] = z[k]*prod(setdiff(inneighbors(g,k),l))
    end
    i_new = (1-β)*(I-α*A).*i + s - s_new
    return θ_new, s_new, i_new, z # must pass z again for easy iteration
end

function simulation_cb(g, α::Real, β::Real, state, tmax::Integer)

end
