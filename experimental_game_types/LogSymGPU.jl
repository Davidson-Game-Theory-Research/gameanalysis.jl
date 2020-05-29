
import Combinatorics: with_replacement_combinations, multinomial
const CwR = with_replacement_combinations

import CUDA
CUDA.allowscalar(false)

include("LogMultinomial.jl")
using .LogMultinomial

include("AbstractGames.jl")

struct LogSymGame_GPU <: SymmetricGame
    num_players::UInt
    num_actions::UInt
    config_table::CUDA.CuArray{Float32,2}
    payoff_table::CUDA.CuArray{Float32,2}
    repeat_table::CUDA.CuArray{Float32,1}
end

function LogSymGame_GPU(num_players, num_actions, payoff_generator)
    num_configs = multinomial(num_players-1, num_actions-1)
    config_table = zeros(Float32, num_configs, num_actions)
    payoff_table = Array{Float32}(undef, num_configs, num_actions)
    repeat_table = Array{Float32}(undef, num_configs)
    for (c,config) in enumerate(CwR(1:num_actions, num_players-1))
        for p in 1:num_players-1
            config_table[c,config[p]] += 1
        end
        repeat_table[c] = logmultinomial(config_table[c,:]...)
        payoff_table[c,:] = payoff_generator(config_table[c,:])
    end
    LogSymGame_GPU(num_players, num_actions, CUDA.cu(config_table),
                   CUDA.cu(payoff_table), CUDA.cu(repeat_table))
end

function deviation_payoffs(game::LogSymGame_GPU, mixed_profile)
    log_prof = CUDA.cu(log.(mixed_profile .+ 1f-40))
    prof_probs = exp.(sum(log_prof.*game.config_table, dims=2).+game.repeat_table)
    Array(sum(game.payoff_table.*prof_probs, dims=1))
end
