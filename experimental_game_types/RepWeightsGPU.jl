import SaferIntegers: SafeUInt64
import Combinatorics: with_replacement_combinations, multinomial
using TensorCast

const CwR = with_replacement_combinations
const MINIMUM_PROBABILITY = 1e-40
const MAXIMUM_PAYOFF = 1e10
const MINIMUM_PAYOFF = 1e-10

struct RepWeightGame_GPU <: SymmetricGame
    num_players::UInt
    num_actions::UInt
    config_table::CUDA.CuArray{Float32,2}
    payoff_table::CUDA.CuArray{Float32,2}
    offset::Float32
    scale::Float32
end

function RepWeightGame_GPU(num_players, num_actions, payoff_generator)
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
    offset = minimum(payoff_table)
    scale = (MAXIMUM_PAYOFF - MINIMUM_PAYOFF) / (maximum(payoff_table) - offset)
    scaled_payoffs = (scale .* (payoff_table .- offset)) .+ MINIMUM_PAYOFF
    weighted_payoffs = log.(scaled_payoffs) .+ repeat_table
    RepWeightGame_GPU(num_players, num_actions, CUDA.cu(config_table),
                      CUDA.cu(weighted_payoffs), offset, 1.0/scale)
end

# function RepWeightGame_GPU(g::SymGame_CPU)
#     RepWeightGame_GPU(g.num_players, g.num_actions, CUDA.cu(g.config_table),
#                       CUDA.cu(log.(g.payoff_table) .+ log.(g.repeat_table)))
# end

function deviation_payoffs(game::RepWeightGame_GPU, mixed_profile)
    log_probs = CUDA.cu(log.(mixed_profile .+ 1f-40)).*game.config_table
    dev_pays = Array(sum(exp.(game.payoff_table.+sum(log_probs, dims=2)), dims=1))
    ((dev_pays .- MINIMUM_PAYOFF) .* game.scale) .+ game.offset
end

# function many_deviation_payoffs(game::RepWeightGame_GPU, mixtures::Array{Float64,2})
#     mixtures = CUDA.cu(log.(mixtures .+ MINIMUM_PROBABILITY))
#     @reduce log_probs[prof,mix] := sum(act) game.config_table[prof,act] * mixtures[mix,act]
#     @reduce dev_pays[mix,act] := sum(prof) exp(game.payoff_table[prof,act] + log_probs[prof,mix])
#     ((dev_pays .- MINIMUM_PAYOFF) .* game.scale) .+ game.offset
# end

function many_deviation_payoffs(game::RepWeightGame_GPU, mixtures::Array{Float64,2})
    mixtures = CUDA.cu(log.(mixtures .+ MINIMUM_PROBABILITY))
    @reduce log_probs[prof,mix] := sum(act) game.config_table[prof,act] * mixtures[mix,act]
    @cast dev_pays[prof,mix,act] := exp(game.payoff_table[prof,act] + log_probs[prof,mix])
    # reshape((sum(dev_pays, dims=1) .- MINIMUM_PAYOFF) .* game.scale .+ game.offset, size(mixtures))
    reshape(sum(dev_pays, dims=1), size(mixtures))
end
