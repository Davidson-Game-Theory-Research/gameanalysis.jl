using SaferIntegers: SafeInt64
using Combinatorics: multinomial, with_replacement_combinations as CwR
using StatsBase: counts

using TensorCast
using CUDA
CUDA.allowscalar(false)

const MAXIMUM_PAYOFF = 1e3
const MINIMUM_PAYOFF = 1e-5

# Stores symmetric games in a dictionary that maps profile vector -> payoff vector
struct PayoffDict <: IterativeGame
    num_players::Int64
    num_actions::Int64
    payoff_table::Dict{Vector{Int64}, Vector{Float64}}
    offset::Float64
    scale::Float64
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

function normalize(payoffs::VecOrMat, offset::Real, scale::Real)
    return scale .* (payoffs .+ offset)
end

function PayoffDict(num_players, num_actions, payoff_generator)
    payoff_table = Dict{Vector{Int64}, Vector{Float64}}()
    min_payoff = Inf
    max_payoff = -Inf
    for (c,config) in enumerate(CwR(1:num_actions, num_players))
        prof = counts(config, 1:num_actions)
        payoffs = payoff_generator(prof)
        payoff_table[prof] = payoffs
        min_payoff = min(minimum(payoffs[prof .> 0]), min_payoff)
        max_payoff = max(maximum(payoffs[prof .> 0]), max_payoff)
    end
    (offset, scale) = set_scale(min_payoff, max_payoff)
    payoff_table = Dict(prof => normalize(pays, offset, scale) for (prof,pays) in payoff_table)
    PayoffDict(num_players, num_actions, payoff_table, offset, scale)
end

function deviation_payoff(game::PayoffDict, mixture::AbstractVector, action::Integer)
    payoff = 0
    for (prof,pays) in game.payoff_table
        if prof[action] > 0
            dev_prof = Array{SafeInt64}(prof)
            dev_prof[action] -= 1
            repeats = multinomial(dev_prof...)
            prob = prod(mixture.^dev_prof)
            payoff += prob * repeats* pays[action]
        end
    end
    return payoff
end

function deviation_payoffs(game::IterativeGame, mixture::AbstractVector)
    return [deviation_payoff(game, mixture, a) for a in 1:game.num_actions]
end


# Same data as PayoffDict, but stored in Arrays to allow vectorization
struct PayoffArrays <: IterativeGame
    num_players::Int64
    num_actions::Int64
    config_table::Array{Int64,2}
    payoff_table::Array{Float64,2}
    offset::Float64
    scale::Float64
end

function PayoffArrays(num_players, num_actions, payoff_generator)
    num_configs = multinomial(num_players, num_actions-1)
    config_table = Array{Int64}(undef, num_configs, num_actions)
    payoff_table = Array{Float64}(undef, num_configs, num_actions)
    min_payoff = Inf
    max_payoff = -Inf
    for (c,config) in enumerate(CwR(1:num_actions, num_players))
        prof = Array{SafeInt64}(counts(config, 1:num_actions))
        config_table[c,:] = prof
        payoffs = payoff_generator(prof)
        payoff_table[c,:] = payoffs
        min_payoff = min(minimum(payoffs[prof .> 0]), min_payoff)
        max_payoff = max(maximum(payoffs[prof .> 0]), max_payoff)
    end
    (offset, scale) = set_scale(min_payoff, max_payoff)
    payoff_table = normalize(payoff_table, offset, scale)
    PayoffArrays(num_players, num_actions, config_table, payoff_table, offset, scale)
end

function deviation_payoff(game::PayoffArrays, mixture::AbstractVector, action::Integer)
    payoff = 0
    dev_configs = copy(game.config_table)
    nonzero = dev_configs[:,action] .> 0
    dev_configs[nonzero,action] .-= 1
    @reduce probs[c] := prod(a) mixture[a].^dev_configs[c,a]
    num_configs = size(dev_configs, 1)
    repeats = zeros(num_configs)
    for c in 1:num_configs
        repeats[c] = multinomial(dev_configs[c,:]...)
    end
    return sum(probs .* repeats .* game.payoff_table[:,action])
end


# Like PayoffArrays, but pre-computes the repetitions
struct RepeatsTable <: SymmetricGame
    num_players::Int64
    num_actions::Int64
    config_table::Array{Int64,2}
    payoff_table::Array{Float64,2}
    repeat_table::Array{Int64,2}
    offset::Float64
    scale::Float64
end

function RepeatsTable(num_players, num_actions, payoff_generator)
    num_configs = multinomial(num_players, num_actions-1)
    config_table = Array{Int64}(undef, num_configs, num_actions)
    payoff_table = Array{Float64}(undef, num_configs, num_actions)
    repeat_table = zeros(Int64, num_configs, num_actions)
    min_payoff = Inf
    max_payoff = -Inf
    for (c,config) in enumerate(CwR(1:num_actions, num_players))
        prof = Array{SafeInt64}(counts(config, 1:num_actions))
        config_table[c,:] = prof
        for a in 1:num_actions
            if prof[a] > 0
                prof[a] -= 1
                repeat_table[c,a] = multinomial(prof...)
                prof[a] += 1
            end
        end
        payoffs = payoff_generator(prof)
        payoff_table[c,:] = payoffs
        min_payoff = min(minimum(payoffs[prof .> 0]), min_payoff)
        max_payoff = max(maximum(payoffs[prof .> 0]), max_payoff)
    end
    (offset, scale) = set_scale(min_payoff, max_payoff)
    payoff_table = normalize(payoff_table, offset, scale)
    RepeatsTable(num_players, num_actions, config_table, payoff_table, repeat_table, offset, scale)
end

function deviation_payoffs(game::RepeatsTable, mixture::AbstractVector)
    mixture = mixture .+ eps(0.0f0)
    @reduce config_probs[c] := prod(a) mixture[a] ^ (game.config_table[c,a])
    @cast weights[c,a] := config_probs[c] * game.repeat_table[c,a] / mixture[a]
    @reduce dev_pays[a] := sum(c) game.payoff_table[c,a] * weights[c,a]
    return dev_pays
end


# Similar pre-computation to RepeatsTable, but storing opponent-profiles
struct DeviationProfiles <: SymmetricGame
    num_players::Int64
    num_actions::Int64
    config_table::Array{Int64,2}
    payoff_table::Array{Float64,2}
    repeat_table::Array{Int64}
    offset::Float64
    scale::Float64
end

function DeviationProfiles(num_players, num_actions, payoff_generator)
    num_configs = multinomial(num_players-1, num_actions-1)
    config_table = Array{Int64}(undef, num_configs, num_actions)
    payoff_table = Array{Float64}(undef, num_configs, num_actions)
    repeat_table = Array{SafeInt64}(undef, num_configs)
    for (c,config) in enumerate(CwR(1:num_actions, num_players-1))
        prof = Array{SafeInt64}(counts(config, 1:num_actions))
        config_table[c,:] = prof
        repeat_table[c] = multinomial(prof...)
        payoff_table[c,:] = payoff_generator(prof)
    end
    (offset, scale) = set_scale(minimum(payoff_table), maximum(payoff_table))
    payoff_table = normalize(payoff_table, offset, scale)
    DeviationProfiles(num_players, num_actions, config_table, payoff_table, repeat_table, offset, scale)
end

function deviation_payoffs(game::DeviationProfiles, mixture::AbstractVector)
    @reduce config_probs[c] := prod(a) mixture[a] ^ game.config_table[c,a]
    @reduce dev_pays[a] := sum(c) game.payoff_table[c,a] * config_probs[c] * game.repeat_table[c]
    return dev_pays
end

# Similar to to DeviationPayoffs, but payoff and repeat tables are combined into one
struct WeightedPayoffs <: SymmetricGame
    num_players::Int64
    num_actions::Int64
    config_table::Array{Int64,2}
    payoff_table::Array{Float64,2}
    offset::Float64
    scale::Float64
end

function WeightedPayoffs(num_players, num_actions, payoff_generator)
    num_configs = multinomial(num_players-1, num_actions-1)
    config_table = Array{Int64}(undef, num_configs, num_actions)
    payoff_table = Array{Float64}(undef, num_configs, num_actions)
    repeat_table = Array{SafeInt64}(undef, num_configs)
    for (c,config) in enumerate(CwR(1:num_actions, num_players-1))
        prof = Array{SafeInt64}(counts(config, 1:num_actions))
        config_table[c,:] = prof
        repeat_table[c] = multinomial(prof...)
        payoff_table[c,:] = payoff_generator(prof)
    end
    (offset, scale) = set_scale(minimum(payoff_table), maximum(payoff_table))
    payoff_table = normalize(payoff_table, offset, scale) .* repeat_table
    WeightedPayoffs(num_players, num_actions, config_table, payoff_table, offset, scale)
end

function deviation_payoffs(game::WeightedPayoffs, mixture::AbstractVector)
    @reduce config_probs[c] := prod(a) mixture[a] ^ game.config_table[c,a]
    @reduce dev_pays[a] := sum(c) game.payoff_table[c,a] * config_probs[c]
    return dev_pays
end


# Similar to WeightedPayoffs, but using a log-transform to turn * into + and ^ into *
struct LogProbabilities <: SymmetricGame
    num_players::Int64
    num_actions::Int64
    config_table::Array{Int64,2}
    payoff_table::Array{Float64,2}
    offset::Float64
    scale::Float64
end

function LogProbabilities(num_players, num_actions, payoff_generator)
    num_configs = multinomial(num_players-1, num_actions-1)
    config_table = Array{Int64}(undef, num_configs, num_actions)
    payoff_table = Array{Float64}(undef, num_configs, num_actions)
    repeat_table = Array{Float64}(undef, num_configs)
    for (c,config) in enumerate(CwR(1:num_actions, num_players-1))
        prof = counts(config, 1:num_actions)
        config_table[c,:] = prof
        repeat_table[c] = logmultinomial(prof...)
        payoff_table[c,:] = payoff_generator(prof)
    end
    (offset, scale) = set_scale(minimum(payoff_table), maximum(payoff_table))
    payoff_table = log.(normalize(payoff_table, offset, scale)) .+ repeat_table
    LogProbabilities(num_players, num_actions, config_table, payoff_table, offset, scale)
end

function deviation_payoffs(game::LogProbabilities, mixture::AbstractVector)
    log_mixture = log.(mixture .+ eps(0e0))
    @reduce log_config_probs[c] := sum(a) log_mixture[a] * game.config_table[c,a]
    @reduce dev_pays[a] := sum(c) exp(game.payoff_table[c,a] + log_config_probs[c])
    return dev_pays
end

function deviation_payoffs(game::LogProbabilities, mixtures::AbstractMatrix)
    log_mixtures = log.(mixtures .+ eps(0e0))
    @reduce log_config_probs[c,m] := sum(a) log_mixtures[a,m] * game.config_table[c,a]
    @reduce dev_pays[a,m] := sum(c) exp(game.payoff_table[c,a] + log_config_probs[c,m])
    return dev_pays
end

function deviation_derivatives(game::LogProbabilities, mixture::AbstractVector)
    mixture = mixture .+ 1e-200
    log_mixture = log.(mixture)
    @reduce log_config_probs[c] := sum(a) log_mixture[a] * game.config_table[c,a]
    @cast deriv_configs[c,a] := game.config_table[c,a] / (mixture[a])
    @reduce dev_jac[a,s] := sum(c) exp(game.payoff_table[c,a] + log_config_probs[c]) * deriv_configs[c,s]
    return dev_jac
end

function deviation_derivatives(game::LogProbabilities, mixtures::AbstractMatrix)
    mixtures = mixtures .+ 1e-200
    log_mixtures = log.(mixtures)
    @reduce log_config_probs[c,m] := sum(a) log_mixtures[a,m] * game.config_table[c,a]
    @cast deriv_configs[c,a,m] := game.config_table[c,a] / (mixtures[a,m])
    @reduce dev_jac[a,s,m] := sum(c) exp(game.payoff_table[c,a] + log_config_probs[c,m]) * deriv_configs[c,s,m]
    return dev_jac
end


# Same as LogProbabilities, but using CUDA arrays
struct GPUArrays <: SymmetricGame
    num_players::Int32
    num_actions::Int32
    config_table::CUDA.CuArray{Float32,2}
    payoff_table::CUDA.CuArray{Float32,2}
    offset::Float32
    scale::Float32
end

function GPUArrays(num_players, num_actions, payoff_generator)
    num_configs = multinomial(num_players-1, num_actions-1)
    config_table = zeros(Float64, num_configs, num_actions)
    payoff_table = Array{Float64}(undef, num_configs, num_actions)
    repeat_table = Array{Float64}(undef, num_configs)
    for (c,config) in enumerate(CwR(1:num_actions, num_players-1))
        prof = counts(config, 1:num_actions)
        config_table[c,:] = prof
        repeat_table[c] = logmultinomial(prof...)
        payoff_table[c,:] = payoff_generator(prof)
    end
    (offset, scale) = set_scale(minimum(payoff_table), maximum(payoff_table))
    payoff_table = log.(normalize(payoff_table, offset, scale)) .+ repeat_table
    GPUArrays(num_players, num_actions, config_table, payoff_table, offset, scale)
end

function deviation_payoffs(game::GPUArrays, mixture::CUDA.CuArray{Float32,1})
    log_mixture = log.(mixture .+ eps(0.0f0))
    @reduce log_config_probs[c] := sum(a) log_mixture[a] * game.config_table[c,a]
    @reduce dev_pays[a] := sum(c) exp(game.payoff_table[c,a] + log_config_probs[c])
    return dev_pays
end

function deviation_payoffs(game::GPUArrays, mixtures::CUDA.CuArray{Float32,2})
    log_mixtures = log.(mixtures .+ eps(0.0f0))
    @reduce log_config_probs[c,m] := sum(a) log_mixtures[a,m] * game.config_table[c,a]
    @reduce dev_pays[a,m] := sum(c) exp(game.payoff_table[c,a] + log_config_probs[c,m])
    return Array(dev_pays)
end

function deviation_payoffs(game::GPUArrays, m::VecOrMat)
    Array(deviation_payoffs(game, CUDA.CuArray{Float32}(m)))
end

function deviation_derivatives(game::GPUArrays, mixture::CUDA.CuArray{Float32,1})
    mixture = mixture .+ 1e-40
    log_mixture = log.(mixture)
    @reduce log_config_probs[c] := sum(a) log_mixture[a] * game.config_table[c,a]
    @cast deriv_configs[c,a] := game.config_table[c,a] / (mixture[a])
    @reduce dev_jac[a,s] := sum(c) exp(game.payoff_table[c,a] + log_config_probs[c]) * deriv_configs[c,s]
    return dev_jac
end

function deviation_derivatives(game::GPUArrays, mixtures::CUDA.CuArray{Float32,2})
    mixtures = mixtures .+ 1e-40
    log_mixtures = log.(mixtures)
    @reduce log_config_probs[c,m] := sum(a) log_mixtures[a,m] * game.config_table[c,a]
    @cast deriv_configs[c,a,m] := game.config_table[c,a] / (mixtures[a,m])
    @reduce dev_jac[a,s,m] := sum(c) exp(game.payoff_table[c,a] + log_config_probs[c,m]) * deriv_configs[c,s,m]
    return dev_jac
end

function deviation_derivatives(game::GPUArrays, m::VecOrMat)
    Array(deviation_derivatives(game, CUDA.CuArray{Float32}(m)))
end


# Same struct as GPUArrays, but data is stored in RAM for measurement with summarysize
# Provides no actual deviation_payoff functionality!
struct CPU_32bit <: SymmetricGame
    num_players::Int32
    num_actions::Int32
    config_table::Array{Float32,2}
    payoff_table::Array{Float32,2}
    offset::Float32
    scale::Float32
end

function CPU_32bit(num_players, num_actions, payoff_generator)
    num_configs = multinomial(num_players-1, num_actions-1)
    config_table = zeros(Float32, num_configs, num_actions)
    payoff_table = Array{Float64}(undef, num_configs, num_actions)
    repeat_table = Array{Float64}(undef, num_configs)
    for (c,config) in enumerate(CwR(1:num_actions, num_players-1))
        prof = counts(config, 1:num_actions)
        config_table[c,:] = prof
        repeat_table[c] = logmultinomial(prof...)
        payoff_table[c,:] = payoff_generator(prof)
    end
    (offset, scale) = set_scale(minimum(payoff_table), maximum(payoff_table))
    payoff_table = log.(normalize(payoff_table, offset, scale)) .+ repeat_table
    CPU_32bit(num_players, num_actions, config_table, payoff_table, offset, scale)
end

# Estimates the size in bytes to store payoffs and configurations
function game_size(GameType, num_players::Integer, num_actions::Integer)
    if GameType == LogProbabilities || GameType == WeightedPayoffs
        config_entries = num_payoffs(num_players, num_actions, dev=true)
        s = config_entries * sizeof(Int64)
        s += config_entries * sizeof(Float64)
    elseif GameType == GPUArrays || GameType == CPU_32bit
        config_entries = num_payoffs(num_players, num_actions, dev=true)
        s = num_configs * sizeof(Float32)
        s += num_configs * sizeof(Float32)
    elseif GameType == DeviationProfiles
        config_entries = num_payoffs(num_players, num_actions, dev=true)
        s = config_entries * sizeof(Int64)
        s += config_entries / num_actions * sizeof(Int64)
        s += config_entries * sizeof(Float64)
    elseif GameType == RepeatsTable
        config_entries = num_payoffs(num_players, num_actions, dev=false)
        s = config_entries * 2 * sizeof(Int64)
        s += config_entries * sizeof(Float64)
    elseif GameType == PayoffArrays
        config_entries = num_payoffs(num_players, num_actions, dev=false)
        s = config_entries * sizeof(Int64)
        s += config_entries * sizeof(Float64)
    elseif GameType == PayoffDict
        config_entries = num_payoffs(num_players, num_actions, dev=false)
        s = config_entries * sizeof(Int64)
        s += config_entries * sizeof(Float64)
        s *= 2 # pessimistic estimate of dictionary overhead
    else
        error("Unrecognized game type: " * string(GameType))
    end
    return s
end