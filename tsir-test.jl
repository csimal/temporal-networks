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

g = activity_driven(N, 100, 0.5, rand(N), 5, 5)
g = random_contacts(binary_tree(7), 100, 0.5, 0.5)
g = random_contacts(watts_strogatz(N, 6, 0.1), 100, 0.5, 0.75)
g = random_contacts(erdos_renyi(N,0.05), 100, 0.5, 0.75)

cl = ContactList(g)
ag = aggregate_network(g)

N = nv(ag)

gplot(ag)

β = 0.1
γ = 0.05

x0 = ones(Int,N)
x0[1] = 2
x0[randperm(N)[1:5]] .= 2
x0 = fill(2, N)

ts, output = event_based_tsir(cl, β, γ, x0)

plot(ts, output[2], label="", linetype=:steppre)

ts_av, out_av, err_av = average_tsir(cl, β, γ, x0, nsims=1000, tmax=100.0)
ts_ib, out_ib = individual_based_continuous(g, β, γ, x0, tmax=100.0)
ts_cb, out_cb = contact_based_continuous(g, β, γ, x0, tmax=100.0, saveat=ts_av)
ts_pa, out_pa = pair_approximation(g, β, γ, x0, tmax=50.0, saveat=ts_av)

cp_ag = ContactProcess(ag, SIR(β,γ))
ts_av_ag, out_av_ag = average(cp_ag, x0, nsims=500, nbins=200, tmax=50.0)
ts_ib_ag, out_ib_ag = meanfield(cp_ag, x0, tmax=50.0)
@time ts_cb_ag, out_cb_ag = contact_based_continuous(cp_ag, x0, 50.0)
@time ts_pa_ag, out_pa_ag = pair_approximation(cp_ag, x0, tmax=30.0)

plot(ts_av, (out_av[2,:])/N,
    linestyle=:dash,
    ylims=(0.0,1.05),
    ylabel="I",
    xlabel="Time",
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

plot(ts_av, log10.(abs.(sum(out_cb[2]-out_pa[2], dims=2))),
    label = "",
    ylabel = "log10(|CB-PA|)",
    xlabel = "Time",
    fontfamily = "Deja Vu"
)

plot(ts_av_ag, out_av_ag[2]/N,
    linestyle=:dash,
    fontfamily="Deja Vu",
    label="Average (static)",
    ylims=(0.0,1.05),
    ylabel="I",
    xlabel="Time"
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
g = activity_driven(N, 100, 1.0, rand(N), 5, 5)
g = random_contacts(watts_strogatz(N, 6, 0.1), 100, 1.0, 0.25)
g = random_contacts(binary_tree(7), 100, 1.0, 0.5)
g = random_contacts(path_graph(50), 100, 1.0, 0.25)
g = random_contacts(star_graph(50), 100, 1.0, 0.75)
g = random_contacts(erdos_renyi(N,0.05), 100, 1.0, 0.75)
ag = aggregate_network(g)
N = nv(ag)


β = 0.8
γ = 0.1

x0 = ones(Int,N)
x0[1] = 2
x0[randperm(N)[1:10]] .= 2
x0 = fill(2,N)

ts, output = event_based_dtsir(g, β, γ, x0)

plot(ts, output[2],
    label = "",
    fontfamily = "Deja Vu",
    linetype = :steppre
)

ts_av, out_av, err_av = average_dtsir(g, β, γ, x0, nsims=1000, tmax = 100.0, nbins=101)
ts_ib, out_ib = individual_based_discrete(g, β, γ, x0, 100)
ts_cb, out_cb = contact_based_discrete(g, β, γ, x0, 100)

scatter(ts_av, out_av[2,:]/N, yerror=err_av[2,:]/N,
    fontfamily = "Deja Vu",
    ylims = (0.0,1.05),
    ylabel = "I",
    xlabel = "Time",
    label = "Average",
    markersize=3,
    lc=:black,
    msc=:black,
    legend = :bottomright
    )
    plot!(ts_ib, sum(out_ib[2], dims=2)/N,
    label="IB",
    marker=:circle,
    markersize=3
    )
    plot!(ts_cb, sum(out_cb[2], dims=2)/N,
    label="CB",
    marker=:circle,
    markersize=3
    )


cp_ag = ContactProcess(ag, SIR(β,γ))
sn_ag = SnapshotList([ag for t in 1:100], 0.0:100.0)

ts_av_ag, out_av_ag, err_av_ag = average_dtsir(sn_ag, β, γ, x0, nsims=1000, tmax=100.0, nbins=101)
ts_ib_ag, out_ib_ag = individual_based_discrete(cp_ag, x0, 100)
ts_cb_ag, out_cb_ag = contact_based_discrete(cp_ag, x0, 100)

ts_ag, out_ag = event_based_dtsir(sn_ag, β, γ, x0)

plot(ts_ag, out_ag[2],
    label = "",
    fontfamily = "Deja Vu",
    linetype = :steppre
)

scatter(ts_av_ag, out_av_ag[2,:]/N, yerror=err_av_ag[2,:]/N,
    fontfamily = "Deja Vu",
    ylims = (0.0,1.05),
    ylabel = "I",
    xlabel = "Time",
    label = "Average",
    markersize = 3,
    lc=:black,
    msc=:black,
    legend = :bottomright
    )
    plot!(ts_ib_ag, sum(out_ib_ag[2], dims=2)/N,
    label="IB",
    marker = :circle,
    markersize = 3
    )
    plot!(ts_cb, sum(out_cb_ag[2], dims=2)/N,
    label="CB",
    marker = :circle,
    markersize = 3
    )
