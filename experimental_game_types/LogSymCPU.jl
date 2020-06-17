import SaferIntegers: SafeUInt64
import Combinatorics: with_replacement_combinations, multinomial

const CwR = with_replacement_combinations


struct LogSymGame_CPU <: SymmetricGame
    num_players::UInt
    num_actions::UInt
    config_table::Array{UInt64,2}
    payoff_table::Array{Float64,2}
    repeat_table::Array{Float64,1}
end

function LogSymGame_CPU(num_players, num_actions, payoff_generator)
    num_configs = multinomial(num_players-1, num_actions-1)
    config_table = zeros(UInt64, num_configs, num_actions)
    payoff_table = Array{Float64}(undef, num_configs, num_actions)
    repeat_table = Array{Float64}(undef, num_configs)
    for (c,config) in enumerate(CwR(1:num_actions, num_players-1))
        for p in 1:num_players-1
            config_table[c,config[p]] += 1
        end
        repeat_table[c] = logmultinomial(config_table[c,:]...)
        payoff_table[c,:] = payoff_generator(config_table[c,:])
    end
    LogSymGame_CPU(num_players, num_actions, config_table, payoff_table, repeat_table)
end

function LogSymGame_CPU(g::SymGame_CPU)
    LogSymGame_CPU(g.num_players, g.num_actions, g.config_table,
                g.payoff_table, log.(g.repeat_table))
end

function deviation_payoffs(game::LogSymGame_CPU, mixed_profile)
    log_probs = log.(mixed_profile).*game.config_table
    replace!(log_probs, NaN=>0)
    config_probs = exp.(sum(log_probs, dims=2) .+ game.repeat_table)
    sum(game.payoff_table.*config_probs, dims=1)
end
