import LinearAlgebra: dot
import Distributions: Normal


abstract type Game end
abstract type SymmetricGame <: Game end
abstract type RoleSymmetricGame <: Game end


function random_game(game_type, players, actions, distribution=Normal(0,1))
    payoff_generator = config -> rand(distribution, size(config))
    game_type(players, actions, payoff_generator)
end


function uniform_mixture(game::SymmetricGame)
    ones(1, game.num_actions) / game.num_actions
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
