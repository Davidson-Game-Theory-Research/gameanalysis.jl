include("to_include.jl")

using BenchmarkTools
using Serialization

player_range = 5:5:50
action_range = 2:6

SymCPU_benchmarks = Array{BenchmarkTools.Trial}(undef, length(player_range), length(action_range))

for (p,players) in enumerate(player_range)
    for (a,actions) in enumerate(action_range)
        println(players, " players, ", actions, " actions")
        g = random_game(SymGame_CPU, players, actions)
        m = uniform_mixture(g)
        benchmark = @benchmark replicator_dynamics($g, $m, 1000)
        SymCPU_benchmarks[p,a] = benchmark
        open(f -> serialize(f, SymCPU_benchmarks), "SymCPU_benchmarks.jls", "w")
    end
end

# SymGPU_benchmarks = Array{BenchmarkTools.Trial}(undef, length(player_range), length(action_range))
#
# for (p,players) in enumerate(player_range)
#     for (a,actions) in enumerate(action_range)
#         println(players, " players, ", actions, " actions")
#         g = random_game(SymGame_GPU, players, actions)
#         m = uniform_mixture(g)
#         benchmark = @benchmark replicator_dynamics($g, $m, 1000)
#         SymGPU_benchmarks[p,a] = benchmark
#         open(f -> serialize(f, SymGPU_benchmarks), "SymGPU_benchmarks.jls", "w")
#     end
# end
#
# LogSymCPU_benchmarks = Array{BenchmarkTools.Trial}(undef, length(player_range), length(action_range))
#
# for (p,players) in enumerate(player_range)
#     for (a,actions) in enumerate(action_range)
#         println(players, " players, ", actions, " actions")
#         g = random_game(LogSymGame_CPU, players, actions)
#         m = uniform_mixture(g)
#         benchmark = @benchmark replicator_dynamics($g, $m, 1000)
#         LogSymCPU_benchmarks[p,a] = benchmark
#         open(f -> serialize(f, LogSymCPU_benchmarks), "LogSymCPU_benchmarks.jls", "w")
#     end
# end
#
# LogSymGPU_benchmarks = Array{BenchmarkTools.Trial}(undef, length(player_range), length(action_range))
#
# for (p,players) in enumerate(player_range)
#     for (a,actions) in enumerate(action_range)
#         println(players, " players, ", actions, " actions")
#         g = random_game(LogSymGame_GPU, players, actions)
#         m = uniform_mixture(g)
#         benchmark = @benchmark replicator_dynamics($g, $m, 1000)
#         LogSymGPU_benchmarks[p,a] = benchmark
#         open(f -> serialize(f, LogSymGPU_benchmarks), "LogSymGPU_benchmarks.jls", "w")
#     end
# end
