using LightGraphs
using TemporalNetworks

@enum EpidemicState susceptible infected recovered

struct EpidemicEvent
    node
    state::EpidemicState
    time::Real
end

function epidemic_step_sir(g::SimpleGraph, α::Real, β::Real, state::Vector{EpidemicState})
    new_state = copy(state)
    for v in shuffle(vertices(g)) # go through nodes in a random order
        if new_state[v] == susceptible
            for u in [i for i in neighbors(g,v) if new_state[i]==infected]
                if rand() < α
                    new_state[v] = infected
                    break
                end
            end
        elseif new_state[v] == infected && rand() < β
            new_state[v] = recovered
        end
    end
    return new_state
end

function epidemic_step_ib(g::SimpleDiGraph, α::Real, β::Real, state)
    (s,i,r) = state # probability vectors for each possible epidemic state
    n = length(s)
    s_new = Vector{Real}(0.0,n)
    i_new = Vector{Real}(0.0,n)
    r_new = Vector{Real}(0.0,n)
    A = adjacency_matrix(g)

    for v in vertices(g)
        s_new[v] = s[v]*prod(u->1-α*A[u,v]*i(u) , 1:n) # TODO: remove v from range
    end
    return (s_new, i_new, r_new)
end
