function replicator_dynamics(game, mixed_profile, iterations)
    offset = minimum_payoff(game)
    for i in 1:iterations
        EVs = deviation_payoffs(game, mixed_profile)
        mixed_profile = normalize_profile(game, mixed_profile .* (EVs.-offset))
    end
    return mixed_profile
end
