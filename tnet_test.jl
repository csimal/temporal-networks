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

#=
(s,i,r,infection,recovery) = montecarlo_sir(g, α, β, state, 500)

s = transpose(hcat(s...))
i = transpose(hcat(i...))
r = transpose(hcat(r...))

recovery = transpose(hcat(recovery...))

plot(i)
=#

s0 = [0.0 1.0 1.0 1.0 1.0]
i0 = [1.0 0.0 0.0 0.0 0.0]
r0 = zeros(5)

state_ib = epidemic_step_ib(g, α, β, (s0, i0, r0))

s_ib = s0'
i_ib = i0'
r_ib = r0'
sk = s0
ik = i0
rk = r0

for j = 1:5
    println(@isdefined(sk))
    #=(sk1, ik1, rk1) = epidemic_step_ib(g, α, β, (sk, ik, rk))
    s_ib = [s_ib sk1']
    i_ib = [i_ib ik1']
    r_ib = [r_ib rk1']
    (sk, ik, rk) = (sk1,ik1,rk1)=#
end
