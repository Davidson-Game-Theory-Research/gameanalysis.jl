import LinearAlgebra: dot
import Distributions: Normal, Dirichlet


abstract type Game end
abstract type SymmetricGame <: Game end
abstract type RoleSymmetricGame <: Game end


function uniform_mixture(game::SymmetricGame)
    ones(game.num_actions) / game.num_actions
end

function random_mixtures(game::SymmetricGame, num_mixtures, α=1)
    if typeof(α) <: Array
        distr = Dirichlet(α)
    else
        distr = Dirichlet(game.num_actions, α)
    end
    rand(distr, num_mixtures)
end

function minimum_payoff(game::SymmetricGame)
    minimum(game.payoff_table)
end

function normalize_profile(game::SymmetricGame, mixed_profile)
    mixed_profile ./ sum(mixed_profile)
end

function regret(game::SymmetricGame, mixed_profile)
    EVs = deviation_payoffs(game, mixed_profile)
    maximum(EVs .- dot(EVs, mixed_profile))
end
