using Distributions: Beta, Uniform
using Combinatorics: combinations

function congestion_game(num_players, num_facilities, facilities_required;
                         max_coefs=[100;10;1.0], beta_params=[1;1])
    actions = collect(combinations(1:num_facilities, facilities_required))
    degr_probs = zeros(length(max_coefs))
    degr_probs[end] = 1
    min_coefs = zeros(length(max_coefs))
    functions = [random_polynomial(; degr_probs=degr_probs, min_coefs=min_coefs, 
                                   max_coefs=max_coefs, beta_params=beta_params)
                                   for f in 1:num_facilities]
    function_inputs = [f in c for f in 1:num_facilities, c in actions]
    action_weights = -function_inputs'
    return SymBAGG(num_players, length(actions), functions, function_inputs, action_weights)
end

function additive_sin_game(num_players, num_actions, num_functions; edge_prob=0.5,
                           weight_range=[-1;1], weight_beta_params=[1;1],
                           poly_params=Dict(), sin_params=[])
    poly_funcs = [random_polynomial(; poly_params...) for _ in 1:num_functions]
    sin_funcs = [random_sin_func(; sin_params...) for _ in 1:num_functions]
    functions = [x -> p(x) .+ s(x) for (p,s) in zip(poly_funcs, sin_funcs)]
    function_inputs = rand(Uniform(0,1), num_functions, num_actions) .> edge_prob
    action_weights = rand(Beta(weight_beta_params...), num_actions, num_functions)
    action_weights .*= weight_range[2] - weight_range[1]
    action_weights .+= weight_range[1]
    SymBAGG(num_players, num_actions, functions, function_inputs, action_weights)
end

function gaussian_mixture_game(num_players, num_actions, num_gaussians; scale_range=[-1e4,1e4],
                               scale_beta_params=[1;1], corr_strength=1, encourage_mixed=true)
    gaussian_params = Dict(
        :mean_scale=>num_players,
        :var_scale=>num_players,
        :covar_scale=>num_players,
        :corr_strength=>corr_strength
    )                               
    gaussians = [ [random_gaussian(Dirichlet(num_actions,.9); gaussian_params...)
                    for g in 1:num_gaussians] for a in 1:num_actions ]
    scales = rand(Beta(scale_beta_params...), num_actions, num_gaussians)
    scales .*= scale_range[2] - scale_range[1]
    scales .+= scale_range[1]
    if encourage_mixed
        gaussian_params[:var_scale] /= 2
        gaussian_params[:covar_scale] /= 2
        α = Vector{Float64}(undef, num_actions)
        for a in 1:num_actions
            fill!(α, eps(0.0))
            α[a] = 1
            gaussians[a][1] = random_gaussian(Dirichlet(α); gaussian_params...)
            scales[a,1] = minimum(scale_range)
        end
    end
    payoffs(config) = [scales[a,:]' * [N(config) for N in gaussians[a]] for a in 1:num_actions]
    return GPUArrays(num_players, num_actions, payoffs)
end

function add_games(g1::SymmetricGame, g2::SymmetricGame; T::DataType=GPUArrays)
    @assert g1.num_players == g2.num_players
    @assert g1.num_actions == g2.num_actions
    return T(g1.num_players, g1.num_actions, s⃗ -> g1.pure_payoffs(s⃗) .+ g2.pure_payoffs(s⃗))
end

function add_games(g1::GPUArrays, g2::GPUArrays; g2_weight::Real=1)
    @assert g1.num_players == g2.num_players
    @assert g1.num_actions == g2.num_actions
    num_configs = size(g1.config_table,1)
    repeat_table = Array{Float64}(undef, num_configs)
    config_table = Array(g1.config_table)
    for c in 1:num_configs
        repeat_table[c] = logmultinomial(config_table[c,:]...)
    end
    payoff_table = denormalize(g1, exp.(Array(g1.payoff_table) .- repeat_table))
    payoff_table .+= denormalize(g2, exp.(Array(g2.payoff_table) .- repeat_table)) .* g2_weight
    (offset, scale) = set_scale(minimum(payoff_table), maximum(payoff_table))
    payoff_table = log.(normalize(payoff_table, offset, scale)) .+ repeat_table
    GPUArrays(g1.num_players, g1.num_actions, config_table, payoff_table, offset, scale)
end