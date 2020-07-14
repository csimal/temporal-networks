using LightGraphs
using DifferentialEquations
using NetworkEpidemics
using Plots

include("ode_models.jl")


N = 100
g = watts_strogatz(N, 10, 0.1)
g = binary_tree(7)
L = ne(g)
N = nv(g)


β = 0.1
γ = 0.05

tmax = 75.0

stoch = ContactProcess(g, SIR(β,γ))

x₀ = ones(Int, N)
x₀[1] = 2

ts, output = gillespie(stoch, x₀, tmax=tmax, nmax=1000000)

plot(ts, output[2]+output[3], linetype=:steppre)

t_av, output_av = average(stoch, x₀, tmax=tmax, nsims=500, nbins=200, nmax=10000)

t_ib, out_ib = meanfield(stoch, x₀, tmax=tmax, saveat=t_av)
t_cb, out_cb = contact_based_continuous(stoch, x₀, tmax, saveat=t_av)
t_pa, out_pa = pair_approximation(stoch, x₀, tmax, saveat=t_av)

plot(t_av, (output_av[2]+output_av[3])/N,
    fontfamily="Deja Vu",
    xlabel="Time",
    ylabel="(I+R)/N",
    label="Average",
    linestyle=:dash,
    legend=:bottomright
    )
plot!(t_ib, sum(out_ib[2]+out_ib[3], dims=2)/N,
    label="IB")
plot!(t_cb, sum(out_cb[2]+out_cb[3], dims=2)/N,
    label="CB")
plot!(t_pa, sum(out_pa[2]+out_pa[3], dims=2)/N,
    label="PA")

plot(t_cb, out_cb[1])
plot(t_pa, out_pa[1], label="")

plot(sol, vars=(2*N+1+L):(2*N+3*L), label="")
