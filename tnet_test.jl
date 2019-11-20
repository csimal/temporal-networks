using LightGraphs
using Plots


cd("C:\\Users\\Dyoz\\temporal networks\\temporal-networks")
#cd("F:\\temporal networks")

include("TemporalNetwork.jl")
include("Epidemics.jl")

#=
f = open("ht09_contact_list.dat")

edges = Vector{Contact}()
for ln in eachline(f)
    vals = map(s->parse(Int,s), split(ln))
    push!(edges, Contact(vals[1]÷20, Edge(vals[2],vals[3])))
end

tnet = TemporalEdgeList(edges)
=#

g = SimpleDiGraph(path_graph(5))
state = [infected, susceptible, susceptible, susceptible, susceptible]

α = 0.5
β = 0.3

#(state2, events) = epidemic_step_sir(g, α, β, state,true)
#states, events = simulation_sir(g, α, β, state)

(s,i,r,infection,recovery) = montecarlo_sir(g, α, β, state, 500)

s = transpose(hcat(s...))
i = transpose(hcat(i...))
r = transpose(hcat(r...))

recovery = transpose(hcat(recovery...))

plot(i)
