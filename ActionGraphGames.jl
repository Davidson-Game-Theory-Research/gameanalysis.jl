using TensorCast

struct SymBAGG <: AbstractSymGame
    num_players::Int64
    num_actions::Int64
    num_functions::Int64
    repetitions::Array{Float64,1}
    function_inputs::Array{Bool,2} # num_functions X num_actions
    action_weights::Array{Float64,2} # num_actions X num_functions
    function_tables::Array{Float64,2}# num_functions X num_players
    GPU::Bool
end

function SymBAGG(num_players, num_actions, functions, function_inputs, action_weights)
    repetitions = logmultinomial.(0:num_players-1, num_players-1:-1:0) # 0...P-1 in, P-1...0 out
    function_tables = [f(c) for f in functions, c in 0:num_players-1]
    SymBAGG(num_players, num_actions, length(functions), repetitions,
            function_inputs, action_weights, function_tables, false)
end

function pure_payoffs(game::SymBAGG, profile::AbstractVector)
    inputs = game.function_inputs * profile
    outputs = [game.function_tables[f,c] for (f,c) in zip(1:game.num_functions, inputs .+ 1)] # +1 because of 1-indexing
    return game.action_weights * outputs
end

# This could be vectorized...
function pure_payoffs(game::SymBAGG, profiles::AbstractMatrix)
    payoffs = Array{Float64,2}(undef, size(profiles)...)
    for p in axes(profiles,2)
        payoffs[:,p] .= pure_payoffs(game, profiles[:,p])
    end
    return payoffs
end

function deviation_payoffs(game::SymBAGG, mixture::AbstractVector)
    in_probs = min.(game.function_inputs * mixture .+ eps(0.0), 1)
    out_probs = 1 .- in_probs .+ eps(0.0)
    in_counts = collect(0:game.num_players - 1)
    out_counts = reverse(in_counts)
    @cast log_probs[f,p] := log(in_probs[f])*in_counts[p] + log(out_probs[f])*out_counts[p] + game.repetitions[p]
    expected_outputs = sum(game.function_tables .* exp.(log_probs), dims=2)
    return game.action_weights * expected_outputs
end

# This could be vectorized...
function deviation_payoffs(game::SymBAGG, mixtures::AbstractMatrix)
    dev_pays = Array{Float64,2}(undef, size(mixtures)...)
    for m in axes(mixtures,2)
        dev_pays[:,m] .= deviation_payoffs(game, mixtures[:,m])
    end
    return dev_pays
end

function deviation_derivatives(game::SymBAGG, mixture::AbstractVector)
    error("Unimplemented: TODO!")
end

function deviation_derivatives(game::SymmetricGame, mixtures::AbstractMatrix)
    error("Unimplemented: TODO!")
end
