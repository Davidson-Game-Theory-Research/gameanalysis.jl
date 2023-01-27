# using Combinatorics: multinomial, with_replacement_combinations as CwR
# using Distributions: Dirichlet
# using StatsBase: counts
using TensorCast

abstract type Game end
abstract type SymmetricGame <: Game end
abstract type FullProfGame <: SymmetricGame end
abstract type IterativeGame <: FullProfGame end


function denormalize(game::SymmetricGame, payoffs::AbstractVecOrMat)
    return payoffs ./ game.scale .- game.offset
end

# Symmetric Games
function num_payoffs(num_players::Integer, num_actions::Integer; dev=true)
    if dev
        return exp(logmultinomial(num_players-1, num_actions-1)) * num_actions
    else
        return exp(logmultinomial(num_players, num_actions-1)) * num_actions
    end
end

# Role Symmetric Games
function num_payoffs(num_players::AbstractVector, num_actions::AbstractVector; dev=true)
    num_roles = length(num_players)
    full_sizes = zeros(num_roles)
    dev_sizes = zeros(num_roles)
    for r in 1:num_roles
        p = num_players[r]
        a = num_actions[r]
        full_sizes[r] = logmultinomial(p, a-1)
        dev_sizes[r] = logmultinomial(p-1, a-1)
    end
    s = sum(full_sizes)
    if dev
        return sum(exp.(s .- full_sizes .+ dev_sizes) .* num_actions)
    else
        return exp(s) * sum(num_actions)
    end
end

function deviation_payoffs(game::SymmetricGame, mixtures::AbstractMatrix)
    dev_pays = Matrix{Float64}(undef, size(mixtures)...)
    for m in 1:size(mixtures,2)
        dev_pays[:,m] .= deviation_payoffs(game, mixtures[:,m])
    end
    return dev_pays
end

function gain_gradients(game::SymmetricGame, mixture::AbstractVector)
    dev_pays = deviation_payoffs(game, mixture)
    mixture_EV = mixture' * dev_pays
    dev_jac = deviation_derivatives(game, mixture)
    util_grads = (mixture' * dev_jac)' .+ dev_pays
    gain_jac = dev_jac .- util_grads'
    gain_jac[dev_pays .< mixture_EV,:] .= 0
    return dropdims(sum(gain_jac, dims=1), dims=1)
end

function gain_gradients(game::SymmetricGame, mixtures::AbstractMatrix)
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

function deviation_gains(game::SymmetricGame, mix::AbstractVecOrMat)
    dev_pays = deviation_payoffs(game, mix)
    dev_pays .- sum(dev_pays .* mix, dims=1)
end

function regret(game::SymmetricGame, mixture::AbstractVector)
    maximum(deviation_gains(game, mixture))
end

function regret(game::SymmetricGame, mixtures::AbstractMatrix)
    dropdims(maximum(deviation_gains(game, mixtures), dims=1), dims=1)
end

# Returns a boolean vector (or matrix if given several mixtures)
# indicating which strategies are best-responses.
function best_responses(game::SymmetricGame, mix::AbstractVecOrMat; atol=eps(0e0))
    dev_pays = deviation_payoffs(game, mix)
    return isapprox.(dev_pays, maximum(dev_pays, dims=1), atol=atol)
end

# The classic function whose fixed-point is Nash.
# For use in Scarf's simplicial subdivision algrotihm.
function better_response(game::SymmetricGame, mix::AbstractVecOrMat; scale_factor::Real=1)
    gains = max.(0,deviation_gains(game, mix)) .* scale_factor
    return (mix .+ gains) ./ (1 .+ sum(gains, dims=1))
end

# for use with ReverseDiff.gradient or similar functions
function optimizable_gain(game::SymmetricGame)
    function dev_gain_sum(mixture::AbstractVector)
        sum(max.(0, deviation_gains(game, mixture)))
    end
end

# throw out mixtures with regret greater than threshold
function filter_regrets(game::SymmetricGame, mixtures::AbstractMatrix; threshold=1e-3, sorted=true)
    mixture_regrets = regret(game, mixtures)
    below_threshold = mixture_regrets .< threshold
    mixtures = mixtures[:,below_threshold]
    mixture_regrets = mixture_regrets[below_threshold]
    if sorted
        mixtures = mixtures[:,sortperm(mixture_regrets)]
    end
    return mixtures
end