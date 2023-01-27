using EllipsisNotation

include("AbstractGames.jl")
include("SimplexOperations.jl")

function replicator_dynamics(game::SymmetricGame, mix::AbstractVecOrMat; iters::Integer=1000, offset::Real=0)
    for i in 1:iters
        mix = simplex_normalize(mix .* (deviation_payoffs(game, mix) .- offset))
    end
    return mix
end

function logged_replicator_dynamics(game::SymmetricGame, mix::AbstractVecOrMat; iters::Integer=1000, offset::Real=0)
    trace = Array{Float64,ndims(mix)+1}(undef, size(mix)..., iters+1)
    for i in 1:iters
        trace[..,i] .= mix
        mix = simplex_normalize(mix .* (deviation_payoffs(game, mix) .- offset))
    end
    trace[..,iters+1] .= mix
    return trace
end

function gain_descent(game::SymmetricGame, mix::AbstractVecOrMat; iters::Integer=1000, step_size::Union{Real,AbstractVector}=1e-6)
    if step_size isa Real
        step_size = [step_size for i in 1:iters]
    end
    for i in 1:iters
        mix = simplex_project(mix .- step_size[i]*gain_gradients(game, mix))
    end
    return mix
end

function logged_gain_descent(game::SymmetricGame, mix::AbstractVecOrMat; iters::Integer=1000, step_size::Union{Real,AbstractVector}=1e-6)
    if step_size isa Real
        step_size = [step_size for i in 1:iters]
    end
    trace = Array{Float64,ndims(mix)+1}(undef, size(mix)..., iters+1)
    for i in 1:iters
        trace[..,i] .= mix
        mix = simplex_project(mix .- step_size[i]*gain_gradients(game, mix))
    end
    trace[..,iters+1] .= mix
    return trace
end

function fictitious_play(game::SymmetricGame, mix::AbstractVecOrMat; iters::Integer=1000, initial_weight::Real=100)
    counts = mix .* initial_weight
    for i in 1:iters
        counts[best_responses(game, mix)] .+= 1
        mix = simplex_normalize(counts)
    end
    return mix
end

function logged_fictitious_play(game::SymmetricGame, mix::AbstractVecOrMat; iters::Integer=1000, initial_weight::Real=100)
    trace = Array{Float64,ndims(mix)+1}(undef, size(mix)..., iters+1)
    trace[..,1] .= mix
    counts = mix .* initial_weight
    for i in 1:iters
        counts[best_responses(game, mix)] .+= 1
        mix = simplex_normalize(counts)
        trace[..,i+1] .= mix
    end
    return trace
end

function iterated_better_response(game::SymmetricGame, mix::AbstractVecOrMat; iters::Integer=1000, step_size::Union{Real,AbstractVector}=1e-6)
    if step_size isa Real
        step_size = [step_size for i in 1:iters]
    end
    for i in 1:iters
        mix = better_response(game, mix, scale_factor=step_size[i])
    end
    return mix
end

function logged_iterated_better_response(game::SymmetricGame, mix::AbstractVecOrMat; iters::Integer=1000, step_size::Union{Real,AbstractVector}=1e-6)
    if step_size isa Real
        step_size = [step_size for i in 1:iters]
    end
    trace = Array{Float64,ndims(mix)+1}(undef, size(mix)..., iters+1)
    for i in 1:iters
        trace[..,i] .= mix
        mix = better_response(game, mix, scale_factor=step_size[i])
    end
    trace[..,iters+1] .= mix
    return trace
end

function batch_nash(nash_func::Function, game::SymmetricGame, mixtures::AbstractMatrix, batch_size::Integer; kwargs...)
    eq_candidates = copy(mixtures)
    num_mixtures = size(mixtures, 2)
    for i in 1:ceil(Int64, num_mixtures ÷ batch_size)
        start_index = (i-1) * batch_size + 1
        stop_index = min(i * batch_size, num_mixtures)
        eq_candidates[:,start_index:stop_index] .= nash_func(game, mixtures[:,start_index:stop_index]; kwargs...)
    end
    return eq_candidates
end