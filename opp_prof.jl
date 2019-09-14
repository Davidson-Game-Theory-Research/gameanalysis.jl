import LinearAlgebra: dot
import Distributions: Normal
import SaferIntegers: SafeUInt64
import CuArrays: CuArray, cu
import Combinatorics: with_replacement_combinations, multinomial
import SpecialFunctions: lfactorial

const CwR = with_replacement_combinations

include("Multinomials.jl")
using .LogMultinomial

struct SymmetricGame_CPU
    num_opponents::UInt
    num_strategies::UInt
    num_profiles::UInt
    profiles::Array{UInt,2}
    payoffs::Array{Float64,2}
    dev_reps::Array{UInt,1}
end

struct LogSymmetricGame_CPU
    num_opponents::UInt
    num_strategies::UInt
    num_profiles::UInt
    profiles::Array{UInt,2}
    payoffs::Array{Float64,2}
    log_dev_reps::Array{Float64,1}
end

struct SymmetricGame_GPU
    num_opponents::UInt
    num_strategies::UInt
    num_profiles::UInt
    profiles::CuArray{Float32,2}
    payoffs::CuArray{Float32,2}
    dev_reps::CuArray{Float32,1}
end

struct LogSymmetricGame_GPU
    num_opponents::UInt
    num_strategies::UInt
    num_profiles::UInt
    profiles::CuArray{Float32,2}
    payoffs::CuArray{Float32,2}
    log_dev_reps::CuArray{Float32,1}
end

function SymmetricGame_CPU(num_players, num_strategies, payoffs)
    num_opponents = num_players - 1
    num_profiles = binomial(num_opponents + num_strategies - 1, num_opponents)
    profiles = zeros(UInt, num_profiles, num_strategies)
    dev_reps = zeros(SafeUInt64, num_profiles)
    for (i,prof) in enumerate(CwR(1:num_strategies, num_opponents))
        for j in 1:num_opponents
            profiles[i,prof[j]] += 1
        end
        dev_reps[i] = multinomial(profiles[i,:]...)
    end
    return SymmetricGame_CPU(num_opponents, num_strategies, num_profiles,
                             profiles, Array{Float64,2}(payoffs), dev_reps)
end

function LogSymmetricGame_CPU(num_players, num_strategies, payoffs)
    num_opponents = num_players - 1
    num_profiles = binomial(num_opponents + num_strategies - 1, num_opponents)
    profiles = zeros(UInt, num_profiles, num_strategies)
    log_dev_reps = zeros(Float64, num_profiles)
    for (i,prof) in enumerate(CwR(1:num_strategies, num_opponents))
        for j in 1:num_opponents
            profiles[i,prof[j]] += 1
        end
        log_dev_reps[i] = lmultinomial(profiles[i,:]...)
    end
    return LogSymmetricGame_CPU(num_opponents, num_strategies, num_profiles,
                                profiles, Array{Float64,2}(payoffs), log_dev_reps)
end

function SymmetricGame_GPU(num_players, num_strategies, payoffs)
    num_opponents = num_players - 1
    num_profiles = binomial(num_opponents + num_strategies - 1, num_opponents)
    profiles = zeros(UInt, num_profiles, num_strategies)
    dev_reps = zeros(SafeUInt64, num_profiles)
    for (i,prof) in enumerate(CwR(1:num_strategies, num_opponents))
        for j in 1:num_opponents
            profiles[i,prof[j]] += 1
        end
        dev_reps[i] = multinomial(profiles[i,:]...)
    end
    return SymmetricGame_GPU(num_opponents, num_strategies, num_profiles,
                             cu(profiles), cu(payoffs), cu(dev_reps))
end

function LogSymmetricGame_GPU(num_players, num_strategies, payoffs)
    num_opponents = num_players - 1
    num_profiles = binomial(num_opponents + num_strategies - 1, num_opponents)
    profiles = zeros(UInt, num_profiles, num_strategies)
    log_dev_reps = zeros(Float64, num_profiles)
    for (i,prof) in enumerate(CwR(1:num_strategies, num_opponents))
        for j in 1:num_opponents
            profiles[i,prof[j]] += 1
        end
        log_dev_reps[i] = lmultinomial(profiles[i,:]...)
    end
    return LogSymmetricGame_GPU(num_opponents, num_strategies, num_profiles,
                                cu(profiles), cu(payoffs), cu(log_dev_reps))
end

function deviation_payoffs(game::SymmetricGame_CPU, mix)
    prof_probs = prod(mix.^game.profiles, dims=2).*game.dev_reps
    return sum(game.payoffs.*prof_probs, dims=1)
end

function deviation_payoffs(game::LogSymmetricGame_CPU, mix)
    prof_probs = exp.(sum(log.(mix).*game.profiles, dims=2).+game.log_dev_reps)
    return sum(game.payoffs.*prof_probs, dims=1)
end

function deviation_payoffs(game::SymmetricGame_GPU, mix)
    prof_probs = prod(cu(mix).^game.profiles, dims=2).*game.dev_reps
    return sum(game.payoffs.*prof_probs, dims=1)
end

function deviation_payoffs(game::LogSymmetricGame_GPU, mix)
    prof_probs = exp.(sum(log.(cu(mix)).*game.profiles, dims=2).+game.log_dev_reps)
    return sum(game.payoffs.*prof_probs, dims=1)
end

function random_game(type, num_players, num_strategies, distribution)
    num_opponents = num_players - 1
    num_profiles = binomial(num_opponents + num_strategies - 1, num_opponents)
    payoffs = rand(distribution, num_profiles, num_strategies)
    return type(num_players, num_strategies, payoffs)
end

function uniform_mixture(num_strategies)
    return ones(num_strategies) / num_strategies
end

function replicator_dynamics(game, mix, iterations)
    offset = minimum(game.payoffs)
    for i in 1:iterations
        EVs = deviation_payoffs(game, mix)
        mix .*= (EVs - offset)
        mix /= sum(mix)
    end
    return mix
end

function regret(game, mix)
    EVs = deviation_payoffs(game, mix)
    return maximum(EVs .- dot(EVs, mix))
end
