

function random_game(game_type, players, actions, distribution=Normal(0,1))
    payoff_generator = config -> rand(distribution, size(config))
    game_type(players, actions, payoff_generator)
end

function gen_test_payoffs(config)
    payoffs = ones(Float64, size(config))
    payoffs[1] = 1e-6
    payoffs[end] = 1e6
    return payoffs
end

function gen_config_payoffs(config)
    payoffs = Array{Float64}(undef, size(config)...)
    payoffs .= config
end
