using LightGraphs
using TemporalNetworks

cd("F:\\temporal networks")
#cd("C:\\Users\\Cedric\\ownCloud\\Julia\\Temporal Networks")

f = open("ht09_contact_list.dat")

edges = Vector{Contact}()
for ln in eachline(f)
    vals = map(s->parse(Int,s), split(ln))
    push!(edges, Contact(vals[1]÷20, Edge(vals[2],vals[3])))
end

tnet = TemporalEdgeList(edges)
