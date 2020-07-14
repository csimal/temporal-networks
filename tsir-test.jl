using LightGraphs
using Plots
using DifferentialEquations
using GraphPlot

include("temporal-network.jl")
include("temporal-sir.jl")
include("random-temporal-networks.jl")
include("ode-models.jl")
include("pair_approximation.jl")

N = 100

g = activity_driven(N, 100, 0.5, rand(N), 3, 5)

g = random_contacts(binary_tree(7), 100, 0.1, 0.75)
g = random_contacts(watts_strogatz(N, 6, 0.1), 100, 0.5, 0.75)
cl = ContactList(g)

ag = aggregate_network(g)

N = nv(ag)

gplot(ag)

β = 0.7
γ = 0.2

x0 = ones(Int,N)
x0[1] = 2

ts, output = event_based_tsir(cl, β, γ, x0)

plot(ts, output[2], label="", linetype=:steppre)

ts_av, μ, σ = average_tsir(cl, β, γ, x0, tmax=30.0)
ts_ib, out_ib = individual_based_continuous(g, β, γ, x0, tmax=30.0)
ts_cb, out_cb = contact_based_continuous(g, β, γ, x0, tmax=30.0)
ts_pa, out_pa = pair_approximation(g, β, γ, x0, tmax=30.0)

cp_ag = ContactProcess(ag, SIR(β,γ))
ts_av_ag, out_av_ag = average(cp_ag, x0, nsims=500, nbins=200, tmax=30.0)
ts_ib_ag, out_ib_ag = meanfield(cp_ag, x0, tmax=30.0)
@time ts_cb_ag, out_cb_ag = contact_based_continuous(cp_ag, x0, 30.0)
@time ts_pa_ag, out_pa_ag = pair_approximation(cp_ag, x0, tmax=30.0)

plot(ts_av, (μ[2,:])/N,
    linestyle=:dash,
    label="Average",
    legend=:topright,
    fontfamily="Deja Vu"
    )
    plot!(ts_ib, sum(out_ib[2], dims=2)/N,
    label="IB"
    )
    plot!(ts_cb, sum(out_cb[2], dims=2)/N,
    label="CB"
    )

plot!(ts_pa, sum(out_pa[2], dims=2)/N,
    label="PA"
    )

plot!(ts_av_ag, out_av_ag[2]/N,
    linestyle=:dash,
    fontfamily="Deja Vu",
    label="Average (static)"
    )
    plot!(ts_ib_ag, sum(out_ib_ag[2], dims=2)/N,
    label="IB (static)"
    )
    plot!(ts_cb_ag, sum(out_cb_ag[2], dims=2)/N,
    label="CB (static)"
    )

plot!(ts_pa_ag, sum(out_pa_ag[2], dims=2)/N,
    label="PA (static)"
    )


plot!(ts_av, (μ[2,:]+σ[2,:])/N,
    linestyle=:dot,
    label=""
    )
    plot!(ts_av, (μ[2,:]-σ[2,:])/N,
    linestyle=:dot,
    label=""
    )


include("discrete-tsir.jl")
include("discrete-time-models.jl")

N = 100
g = random_contacts(watts_strogatz(N, 6, 0.1), 100, 1.0, 0.75)
g = random_contacts(binary_tree(7), 100, 1.0, 0.75)
g = random_contacts(path_graph(50), 100, 1.0, 0.75)
g = random_contacts(star_graph(50), 100, 1.0, 0.75)
ag = aggregate_network(g)
N = nv(ag)

β = 0.8
γ = 0.01

x0 = ones(Int,N)
x0[1] = 2
x0 .= 2

ts, output = event_based_dtsir(g, β, γ, x0)

plot(ts, output[2],
    label = "",
    fontfamily = "Deja Vu",
    linetype = :steppre
)

ts_av, out_av, err_av = average_dtsir(g, β, γ, x0, nsims=500, tmax = 60.0, nbins=61)
ts_ib, out_ib = individual_based_discrete(g, β, γ, x0, 60)
ts_cb, out_cb = contact_based_discrete(g, β, γ, x0, 60)

scatter(ts_av, out_av[2,:]/N, yerror=err_av[2,:]/N,
    fontfamily = "Deja Vu",
    label = "Average",
    )
    plot!(ts_ib, sum(out_ib[2], dims=2)/N,
    label="IB"
    )
    plot!(ts_cb, sum(out_cb[2], dims=2)/N,
    label="CB"
    )


cp_ag = ContactProcess(ag, SIR(β,γ))
sn_ag = SnapshotList([ag for t in 1:100], 0.0:100.0)

ts_av_ag, out_av_ag, err_av_ag = average_dtsir(sn_ag, β, γ, x0, nsims=1000, tmax=60.0, nbins=61)
ts_ib_ag, out_ib_ag = individual_based_discrete(cp_ag, x0, 60)
ts_cb_ag, out_cb_ag = contact_based_discrete(cp_ag, x0, 60)

ts_ag, out_ag = event_based_dtsir(sn_ag, β, γ, x0)

plot(ts_ag, out_ag[2],
    label = "",
    fontfamily = "Deja Vu",
    linetype = :steppre
)

scatter(ts_av_ag, out_av_ag[2,:]/N, yerror=err_av_ag[2,:]/N,
    fontfamily = "Deja Vu",
    label = "Average",
    )
    plot!(ts_ib_ag, sum(out_ib_ag[2], dims=2)/N,
    label="IB"
    )
    plot!(ts_cb, sum(out_cb_ag[2], dims=2)/N,
    label="CB"
    )
