import SaferIntegers: SafeUInt64
import Combinatorics: with_replacement_combinations, multinomial
import TensorCast: @reduce

const CwR = with_replacement_combinations


struct RepWeightGame_CPU <: SymmetricGame
    num_players::UInt
    num_actions::UInt
    config_table::Array{UInt64,2}
    payoff_table::Array{Float64,2}
    offset::Float64
end

function RepWeightGame_CPU(num_players, num_actions, payoff_generator)
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
    offset = minimum(payoff_table) - 1
    weighted_payoffs = log.(payoff_table .- offset) .+ repeat_table
    RepWeightGame_CPU(num_players, num_actions, config_table, weighted_payoffs, offset)
end

function RepWeightGame_CPU(g::SymGame_CPU)
    RepWeightGame_CPU(g.num_players, g.num_actions, g.config_table,
                      log.(g.payoff_table) .+ log.(g.repeat_table))
end

function deviation_payoffs(game::RepWeightGame_CPU, mixed_profile)
    log_probs = log.(mixed_profile).*game.config_table
    replace!(log_probs, NaN=>0)
    sum(exp.(game.payoff_table.+sum(log_probs, dims=2)), dims=1) .+ game.offset
end

function many_deviation_payoffs(game::RepWeightGame_CPU, mixtures::Array{Float64,2})
    mixtures = log.(mixtures .+ 1e-40)
    @reduce log_probs[prof,mix] := sum(act) game.config_table[prof,act] * mixtures[mix,act]
    @reduce dev_pays[mix,act] := sum(prof) exp(game.payoff_table[prof,act] + log_probs[prof,mix])
    dev_pays .+ game.offset
end
