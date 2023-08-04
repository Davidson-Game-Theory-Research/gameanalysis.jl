using Pkg
Pkg.activate("../..")

using DataFrames, CSV
using Plots, LaTeXStrings

ENV["GKSwstype"] = "100" # allows running over SSH

PayoffDict_df = DataFrame(CSV.File("../data/PayoffDict_dev_pays_timing.csv"))
PayoffArrays_df = DataFrame(CSV.File("../data/PayoffArrays_dev_pays_timing.csv"))
RepeatsTable_df = DataFrame(CSV.File("../data/RepeatsTable_dev_pays_timing.csv"))
DeviationProfiles_df = DataFrame(CSV.File("../data/DeviationProfiles_dev_pays_timing.csv"))
WeightedPayoffs_df = DataFrame(CSV.File("../data/WeightedPayoffs_dev_pays_timing.csv"))
LogProbabilities_df = DataFrame(CSV.File("../data/LogProbabilities_dev_pays_timing.csv"))
GPUArrays_df = DataFrame(CSV.File("../data/GPUArrays_dev_pays_timing.csv"))

plot(xaxis=:log, yaxis=:log, legend=(0.1,0.96), legend_font_pointsize=7);
plot!(xlabel=L"Number of Players, $P$ (log scale)", ylabel="Time in Seconds (log scale)");#, title="Time to Compute Deviation Payoffs")

a = 4
plot_kwds = Dict(:markershape=>:circle, :markersize=>3, :linewidth=>2);
PayoffDict_times =  sort(PayoffDict_df[PayoffDict_df.actions .== a, :]);
plot!(PayoffDict_times.players, PayoffDict_times.min_time, label="Payoff Dictionary"; plot_kwds...);
PayoffArrays_times =  sort(PayoffArrays_df[PayoffArrays_df.actions .== a, :]);
plot!(PayoffArrays_times.players, PayoffArrays_times.min_time, label="Array Vectorization"; plot_kwds...);
RepeatsTable_times =  sort(RepeatsTable_df[RepeatsTable_df.actions .== a, :]);
plot!(RepeatsTable_times.players, RepeatsTable_times.min_time, label="Pre-Computing Reps."; plot_kwds...);
DeviationProfiles_times =  sort(DeviationProfiles_df[DeviationProfiles_df.actions .== a, :]);
plot!(DeviationProfiles_times.players, DeviationProfiles_times.min_time, label="Opponent Configs"; plot_kwds...);
WeightedPayoffs_times =  sort(WeightedPayoffs_df[WeightedPayoffs_df.actions .== a, :]);
plot!(WeightedPayoffs_times.players, WeightedPayoffs_times.min_time, label="Pre-Weighting by Reps."; plot_kwds...);
LogProbabilities_times =  sort(LogProbabilities_df[LogProbabilities_df.actions .== a .&& LogProbabilities_df.batch_size .== 1, :]);
plot!(LogProbabilities_times.players, LogProbabilities_times.min_time, label="Log Transform"; plot_kwds...);
GPUArrays_times =  sort(GPUArrays_df[GPUArrays_df.actions .== a .&& GPUArrays_df.batch_size .== 1, :]);
plot!(GPUArrays_times.players, GPUArrays_times.min_time, label="GPU Acceleration"; plot_kwds...);
Batch64_times =  sort(GPUArrays_df[GPUArrays_df.actions .== a .&& GPUArrays_df.batch_size .== 64, :]);
plot!(Batch64_times.players, Batch64_times.min_time, label="Batch Processing: 64"; plot_kwds...);

annotate!(250, 0.025, text(L"A = 4", 12));
annotate!(250, 0.005, text(L"1024\;\textrm{mixtures}", 12));
ylims!((10^-3,10^3.5));
plot!(yticks=([10.0^i for i=-3:3],[.001, .01, .1, 1, 10, 100, 1000]));
xlims!((3.9,394));
plot!(xticks=([2^i for i=2:9],[2^i for i=2:9]));

savefig("deviation_payoff_timing_4A.pdf");