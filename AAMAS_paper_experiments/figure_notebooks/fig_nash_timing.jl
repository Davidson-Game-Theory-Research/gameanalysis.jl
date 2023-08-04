using Pkg
Pkg.activate("../..")

using DataFrames, CSV
using Plots, LaTeXStrings

PayoffDict_RD_df = DataFrame(CSV.File("../data/PayoffDict_RD_timing.csv"))
GPUArrays_RD_df = DataFrame(CSV.File("../data/GPUArrays_RD_timing.csv"))
GPUArrays_GD_df = DataFrame(CSV.File("../data/GPUArrays_GD_timing.csv"))

ENV["GKSwstype"] = "100" # allows running over SSH

a = 4
plot_kwds = Dict(:markershape=>:circle, :markersize=>3, :linewidth=>2)
plot(xaxis=:log, yaxis=:log, legend=(0.77,0.30), legend_font_pointsize=6)
plot!(xlabel=L"Number of Players, $P$ (log scale)", ylabel="Time in Seconds (log scale)")


PayoffDict_RD_times =  sort(PayoffDict_RD_df[PayoffDict_RD_df.actions .== a, :])
plot!(PayoffDict_RD_times.players, PayoffDict_RD_times.min ./ 10^9, label="Prior Data Structure\n+ Replicator Dynamics\n"; plot_kwds...)

GPUArrays_RD_times =  sort(GPUArrays_RD_df[GPUArrays_RD_df.actions .== a, :])
plot!(GPUArrays_RD_times.players, GPUArrays_RD_times.min ./ 10^9, label="Our Data Structure\n+ Replicator Dynamics\n"; plot_kwds...)

GPUArrays_GD_times =  sort(GPUArrays_GD_df[GPUArrays_GD_df.actions .== a, :])
plot!(GPUArrays_GD_times.players, GPUArrays_GD_times.min ./ 10^9, label="Our Data Structure\n+ Gradient Descent"; plot_kwds...)

annotate!(2.7, 2000, text(L"A = 4", 12))
annotate!(5, 500, text(L"100\;\textrm{starting\;mixtures}", 12))
ylims!((10^-2,10^4))
plot!(yticks=([10.0^i for i=-2:4],["0.01", "0.1", "1", "10", "100", "1,000", "10,000"]))
xlims!((1.92,530))
plot!(xticks=([2^i for i=1:9],[2^i for i=1:9]))

savefig("nash_timing_4A.pdf")