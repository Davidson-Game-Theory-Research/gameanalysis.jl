

function game_sizes(num_players, num_strategies)
    num_roles = length(num_players)
    full_sizes = zeros(num_roles)
    dev_sizes = zeros(num_roles)
    for r in 1:num_roles
        p = num_players[r]
        s = num_strategies[r]
        full_sizes[r] = binomial(p+s-1, p)
        dev_sizes[r] = binomial(p+s-2, p-1)
    end
    monolithic_size = prod(full_sizes) * sum(num_strategies)
    separated_size = 0
    for r in 1:num_roles
        separated_size += prod(full_sizes) / full_sizes[r] * dev_sizes[r] * num_strategies[r]
    end
    return monolithic_size, separated_size
end