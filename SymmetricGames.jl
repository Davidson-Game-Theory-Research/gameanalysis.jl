using Combinatorics: multinomial, with_replacement_combinations as CwR
using StatsBase: counts

using TensorCast
import CUDA
CUDA.allowscalar(false)

const MAXIMUM_PAYOFF = 1e5
const MINIMUM_PAYOFF = 1e-5
const F32_EPSILON = eps(1f-20)
const F64_EPSILON = eps(1e-40)

abstract type AbstractSymGame end

struct SymmetricGame <: AbstractSymGame
    num_players::Integer
    num_actions::Integer
    config_table::AbstractMatrix
    payoff_table::AbstractMatrix
    offset::AbstractFloat
    scale::AbstractFloat
    ε::AbstractFloat
end

function SymmetricGame(num_players, num_actions, payoff_generator; GPU=false)
    num_configs = multinomial(num_players-1, num_actions-1)
    config_table = zeros(Float64, num_actions, num_configs)
    payoff_table = Array{Float64}(undef, num_actions, num_configs)
    repeat_table = Array{Float64}(undef, 1, num_configs)
    for (c,config) in enumerate(CwR(1:num_actions, num_players-1))
        prof = counts(config, 1:num_actions)
        config_table[:,c] = prof
        repeat_table[c] = logmultinomial(prof...)
        payoff_table[:,c] = payoff_generator(prof)
    end
    (offset, scale) = set_scale(minimum(payoff_table), maximum(payoff_table))
    payoff_table = log.(normalize(payoff_table, offset, scale)) .+ repeat_table
    if GPU
        config_table = CUDA.CuArray{Float32,2}(config_table)
        payoff_table = CUDA.CuArray{Float32,2}(payoff_table)
        num_players = Int32(num_players)
        num_actions = Int32(num_actions)
        offset = Float32(offset)
        scale = Float32(scale)
        ε = F32_EPSILON
    else
        num_players = Int64(num_players)
        num_actions = Int64(num_actions)
        offset = Float64(offset)
        scale = Float64(scale)
        ε = F64_EPSILON
    end
    SymmetricGame(num_players, num_actions, config_table, payoff_table, offset, scale, ε)
end

function set_scale(min_payoff, max_payoff)
    scale = (MAXIMUM_PAYOFF - MINIMUM_PAYOFF) / (max_payoff - min_payoff)
    if !isfinite(scale)
        scale = 1
    end
    offset = MINIMUM_PAYOFF / scale - min_payoff
    if !isfinite(offset)
        offset = 0
    end
    return (offset, scale)
end

function normalize(payoffs::AbstractVecOrMat, offset::Real, scale::Real)
    return scale .* (payoffs .+ offset)
end

function denormalize(payoffs::AbstractVecOrMat, offset::Real, scale::Real)
    return (payoffs ./ scale) .- offset
end

function pure_payoffs(game::SymmetricGame, profile)
    error("Unimplemented! Requires: ranking algorithm for combinations-with-replacement.")
end

function deviation_payoffs(game::SymmetricGame, mixture::AbstractVector)
    log_mixture = log.(mixture .+ game.ε)
    @reduce log_config_probs[c] := sum(a) log_mixture[a] * game.config_table[a,c]
    @reduce dev_pays[a] := sum(c) exp(game.payoff_table[a,c] + log_config_probs[c])
    return dev_pays
end

function deviation_payoffs(game::SymmetricGame, mixtures::AbstractMatrix)
    log_mixtures = log.(mixtures .+ game.ε)
    @reduce log_config_probs[m,c] := sum(a) log_mixtures[a,m] * game.config_table[a,c]
    @reduce dev_pays[a,m] := sum(c) exp(game.payoff_table[a,c] + log_config_probs[m,c])
    return dev_pays
end

function deviation_derivatives(game::SymmetricGame, mixture::AbstractVector)
    mixture = mixture .+ game.ε
    log_mixture = log.(mixture)
    @reduce log_config_probs[c] := sum(a) log_mixture[a] * game.config_table[a,c]
    @cast deriv_configs[a,c] := game.config_table[a,c] / (mixture[a])
    @reduce dev_jac[a,s] := sum(c) exp(game.payoff_table[a,c] + log_config_probs[c]) * deriv_configs[s,c]
    return dev_jac
end

function deviation_derivatives(game::SymmetricGame, mixtures::AbstractMatrix)
    mixtures = mixtures .+ game.ε
    log_mixtures = log.(mixtures)
    @reduce log_config_probs[m,c] := sum(a) log_mixtures[a,m] * game.config_table[a,c]
    @cast deriv_configs[a,m,c] := game.config_table[a,c] / (mixtures[a,m])
    @reduce dev_jac[a,s,m] := sum(c) exp(game.payoff_table[a,c] + log_config_probs[m,c]) * deriv_configs[s,m,c]
    return dev_jac
end

function deviation_gains(game::AbstractSymGame, mix::AbstractVecOrMat)
    dev_pays = deviation_payoffs(game, mix)
    max.(dev_pays .- sum(dev_pays .* mix, dims=1), 0)
end

function gain_gradients(game::AbstractSymGame, mixture::AbstractVector)
    dev_pays = deviation_payoffs(game, mixture)
    mixture_EV = mixture' * dev_pays
    dev_jac = deviation_derivatives(game, mixture)
    util_grads = (mixture' * dev_jac)' .+ dev_pays
    gain_jac = dev_jac .- util_grads'
    gain_jac[dev_pays .< mixture_EV,:] .= 0
    return dropdims(sum(gain_jac, dims=1), dims=1)
end

function gain_gradients(game::AbstractSymGame, mixtures::AbstractMatrix)
    dev_pays = deviation_payoffs(game, mixtures)
    @reduce mixture_expectations[m] := sum(a) mixtures[a,m] * dev_pays[a,m]
    dev_jac = deviation_derivatives(game, mixtures)
    @reduce util_grads[s,m] := sum(a) mixtures[a,m] * dev_jac[a,s,m]
    util_grads .+= dev_pays
    @cast gain_jac[s,a,m] := dev_jac[a,s,m] - util_grads[s,m]
    # The findall shouldn't be necessary here; this is a Julia language bug.
    # See discourse.julialang.org/t/slicing-and-boolean-indexing-in-multidimensional-arrays
    gain_jac[:,findall(dev_pays .< mixture_expectations')] .= 0
    return dropdims(sum(gain_jac, dims=2), dims=2)
end

function regret(game::AbstractSymGame, mixture::AbstractVector)
    maximum(deviation_gains(game, mixture))
end

function regret(game::AbstractSymGame, mixtures::AbstractMatrix)
    dropdims(maximum(deviation_gains(game, mixtures), dims=1), dims=1)
end

# Returns a boolean vector (or matrix if given several mixtures)
# indicating which strategies are best-responses.
function best_responses(game::AbstractSymGame, mix::AbstractVecOrMat; atol=eps(0e0))
    dev_pays = deviation_payoffs(game, mix)
    return isapprox.(dev_pays, maximum(dev_pays, dims=1), atol=atol)
end

# The classic function whose fixed-point is Nash.
# For use in Scarf's simplicial subdivision algrotihm.
function better_response(game::AbstractSymGame, mix::AbstractVecOrMat; scale_factor::Real=1)
    gains = max.(0,deviation_gains(game, mix)) .* scale_factor
    return (mix .+ gains) ./ (1 .+ sum(gains, dims=1))
end

# throw out mixtures with regret greater than threshold
function filter_regrets(game::AbstractSymGame, mixtures::AbstractMatrix; threshold=1e-3, sorted=true)
    mixture_regrets = regret(game, mixtures)
    below_threshold = mixture_regrets .< threshold
    mixtures = mixtures[:,below_threshold]
    mixture_regrets = mixture_regrets[below_threshold]
    if sorted
        mixtures = mixtures[:,sortperm(mixture_regrets)]
    end
    return mixtures
end