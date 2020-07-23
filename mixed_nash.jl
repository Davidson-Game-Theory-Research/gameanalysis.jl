function replicator_dynamics(game, mixed_profile, iterations)
    for i in 1:iterations
        EVs = deviation_payoffs(game, mixed_profile)
        weights = mixed_profile .* (EVs .- game.offset .+ MINIMUM_PAYOFF)
        mixed_profile = weights ./ sum(weights)
    end
    return mixed_profile
end

function RD_random_restarts(game, num_mixtures, iterations)
    starting_points = random_mixtures(game, num_mixtures)
    equilibria = Array{Float64}(undef, size(starting_points))
    for m in 1:num_mixtures
        mix = reshape(starting_points[m,:], (1,game.num_actions))
        equilibria[m,:] = replicator_dynamics(game, mix, iterations)
    end
    return equilibria
end

#assumes the game is already normalized
function parallel_replicator_dynamics(game, mixed_profiles, iterations)
    for i in 1:iterations
        EVs = many_deviation_payoffs(game, mixed_profiles)
        weights = mixed_profiles .* EVs
        mixed_profiles = weights ./ sum(weights, dims=2)
    end
    return mixed_profiles
end

function RD_trace(game, mixed_profile, iterations)
    trace = Array{Float64}(undef, iterations + 1, length(mixed_profile))
    trace[1,:] = mixed_profile
    for i in 1:iterations
        EVs = deviation_payoffs(game, mixed_profile)
        weights = mixed_profile .* (EVs .- game.offset .+ MINIMUM_PAYOFF)
        mixed_profile = weights ./ sum(weights)
        trace[i+1,:] = mixed_profile
    end
    return trace
end
