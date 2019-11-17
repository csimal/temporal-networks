using LightGraphs
using TemporalNetworks
using SparseArrays

@enum EpidemicState susceptible infected recovered

struct EpidemicEvent
    node
    state::EpidemicState
    time::Real
end


function epidemic_step_sir(g::SimpleDiGraph, α::Real, β::Real, state::Vector{EpidemicState})
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
        s_new[v] = s[v]*prod(u->1-α*i(u) , filter(i->A[i,v]!=0.0, 1:n))
        i_new[v] = (1-β)*i[v] + s[v] - s_new[v]
        r_new[v] = 1 - s_new[v] - i_new[v]
    end
    return (s_new, i_new, r_new)
end

function epidemic_step_cb(g::SimpleDiGraph, α::Real, β::Real, state)
    (θ,s,i,r) = state # NB. the state is described here by edge quantities contained in square matrices
    n = size(θ,1)
    θ_new = spzeros(n,n) # empty sparse matrices
    s_new = spzeros(n,n)
    i_new = spzeros(n,n)
    r_new = spzeros(n,n)
    A = adjacency_matrix(g)

    for u in vertices(g)
        for v in vertices(g)
            
        end
    end
end
