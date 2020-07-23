import SaferIntegers: SafeUInt64
import Combinatorics: with_replacement_combinations, multinomial
using TensorCast
import LinearAlgebra: dot

const CwR = with_replacement_combinations
const MINIMUM_PROBABILITY = 1e-40
const MAXIMUM_PAYOFF = 1.0
const MINIMUM_PAYOFF = 1e-10


struct RepWeightGame_CPU <: SymmetricGame
    num_players::UInt
    num_actions::UInt
    config_table::Array{UInt64,2}
    payoff_table::Array{Float64,2}
    offset::Float64
    scale::Float64
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
    offset = minimum(payoff_table)
    scale = (MAXIMUM_PAYOFF - MINIMUM_PAYOFF) / (maximum(payoff_table) - offset)
    scaled_payoffs = (scale .* (payoff_table .- offset)) .+ MINIMUM_PAYOFF
    weighted_payoffs = log.(scaled_payoffs) .+ repeat_table
    RepWeightGame_CPU(num_players, num_actions, config_table,
                      weighted_payoffs, offset, 1.0/scale)
end

#de-normalized
function deviation_payoffs(game::RepWeightGame_CPU, mixed_profile)
    log_probs = log.(mixed_profile .+ MINIMUM_PROBABILITY).*game.config_table
    dev_pays = sum(exp.(game.payoff_table.+sum(log_probs, dims=2)), dims=1)
    #((dev_pays .- MINIMUM_PAYOFF) .* game.scale) .+ game.offset
end

#kept normalized
function many_deviation_payoffs(game::RepWeightGame_CPU, mixtures::Array{Float64,2})
    mixtures = log.(mixtures .+ MINIMUM_PROBABILITY)
    @reduce log_probs[prof,mix] := sum(act) game.config_table[prof,act] * mixtures[mix,act]
    @reduce dev_pays[mix,act] := sum(prof) exp(game.payoff_table[prof,act] + log_probs[prof,mix])
end

function deviation_jacobian(game::RepWeightGame_CPU, mixed_profile)
    mixed_profile = mixed_profile .+ MINIMUM_PROBABILITY
    log_probs = sum(log.(mixed_profile) .* game.config_table, dims=2)
    deriv_configs = game.config_table ./ mixed_profile
    @reduce dj[a,s] := sum(p) exp(game.payoff_table[p,a] + log_probs[p]) * deriv_configs[p,s]
end

function gain_gradient(game::RepWeightGame_CPU, mixed_profile)
    mp = mixed_profile[1,:] # MAKE PROFILES VECTORS!!!!!!!!!!!!!!!!!!
    dev_pays = deviation_payoffs(game, mixed_profile)[1,:]
    mixture_expectation = dot(mixed_profile, dev_pays)
    dev_jac = deviation_jacobian(game, mixed_profile)
    @reduce util_grad[s] := sum(a)  mp[a] * dev_jac[a,s]
    util_grad .+= dev_pays
    @cast gain_jac[a,s] := dev_jac[a,s] - util_grad[s]
    gain_jac[dev_pays .< mixture_expectation,:] .= 0
    total_gain_grad = sum(gain_jac, dims=1)
end

function gain_function(game::RepWeightGame_CPU)
    function total_gain(mixed_profile)
        dev_pays = deviation_payoffs(game, mixed_profile)
        sum(max.(0, dev_pays .- dot(dev_pays, mixed_profile)))
    end
end

function gain_squared_function(game::RepWeightGame_CPU)
    function total_gain(mixed_profile)
        dev_pays = deviation_payoffs(game, mixed_profile)
        gains = max.(0, dev_pays .- dot(dev_pays, mixed_profile))
        sum(gains.^2)
    end
end
