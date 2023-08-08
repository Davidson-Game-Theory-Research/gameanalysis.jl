using Distributions: Dirichlet, Multinomial
using Combinatorics: multinomial, with_replacement_combinations as CwR
using StatsBase: counts

include("LogMultinomial.jl")

function uniform_mixture(num_actions::Integer)
    ones(num_actions) / num_actions
end

function random_mixtures(num_actions::Integer, num_mixtures::Integer, α=1; GPU=false)
    if α isa Array
        distr = Dirichlet(α)
    else
        distr = Dirichlet(num_actions, α)
    end
    mixtures = rand(distr, num_mixtures)
    if GPU
        return CUDA.CuArray{Float32,2}(mixtures)
    else
        return mixtures
    end
end

# samples one random profile from a given mixture
function sample_profile(num_players::Integer, mixture::AbstractVector)
    return rand(Multinomial(num_players, mixture))
end

# samples one random profile from each of the given mixtures
function sample_profiles(num_players::Integer, mixtures::AbstractMatrix)
    profiles = Matrix{Int64}(undef, size(mixtures))
    for m in axes(mixtures,2)
        profiles[:,m] .= sample_profile(num_players, mixtures[:,m])
    end
    return profiles
end

function num_profiles(num_players::Integer, num_actions::Integer)
    multinomial(num_players-1, num_actions-1)
end

# creates a grid of equally spaced points throughout the simplex
function mixture_grid(num_actions::Integer, points_per_dim::Integer)
    num_mixtures = multinomial(num_actions-1, points_per_dim-1)
    mixtures = Array{Float64,2}(undef, num_actions, num_mixtures)
    for (m,config) in enumerate(CwR(1:num_actions, points_per_dim-1))
        mix = counts(config, 1:num_actions) ./ (points_per_dim-1)
        mixtures[:,m] = mix
    end
    return mixtures
end

# creates a grid of equally spaced points within the sub-simplex specified by corners
function mixture_grid(corners::AbstractMatrix, points_per_dim::Integer)
    grid = mixture_grid(size(corners,2), points_per_dim)
    return corners * grid
end

# identifies the largest mixture_grid with at most max_mixtures elements
function finest_grid(num_actions::Integer, max_mixtures::Integer)
    points_per_dim = 2
    num_mixtures = num_actions
    for p in 3:max_mixtures
        m = multinomial(num_actions-1, p-1)
        if m > max_mixtures
            break
        else
            num_mixtures = m
            points_per_dim = p
        end
    end
    return (points_per_dim, num_mixtures)
end

# creates a grid of equally spaced points surrounding a given midpoint
function grid_around(midpoint::AbstractVector, scale::AbstractFloat, points_per_dim::Integer)
    num_actions = size(midpoint,1)
    offsets = (mixture_grid(num_actions, points_per_dim) .- 1/3) .* scale
    midpoint = simplex_project(midpoint, 1-scale)
    return midpoint .+ offsets
end

# creates combined grids surrounding several midpoints given by the array's columns
function grid_around(midpoints::AbstractMatrix, scale::AbstractFloat, points_per_dim::Integer; prevent_overlap=false)
    num_actions = size(midpoints,1)
    num_grids = size(midpoints,2)
    mixtures_per_grid = multinomial(num_actions-1, points_per_dim-1)
    grids = Matrix{Float64}(undef, num_actions, mixtures_per_grid * num_grids)
    for g in 1:num_grids
        grids[:,(g-1)*mixtures_per_grid+1:g*mixtures_per_grid] .= grid_around(midpoints[:,g], scale, points_per_dim)
    end
    if prevent_overlap
        return unique(grids, dims=2)
    else
        return grids
    end
end

# notation follows https://arxiv.org/pdf/1309.1541.pdf
function simplex_project(y⃗::AbstractVector)
    D = length(y⃗)
    u⃗  = sort(y⃗, rev=true)
    λ⃗ = (1 .- cumsum(u⃗)) ./ (1:D)
    λ = λ⃗[findlast(u⃗ .+ λ⃗ .> 0)]
    max.(y⃗ .+ λ, 0)
end

# version that vectorizes over the columns of M
function simplex_project(M::AbstractMatrix)
    D = size(M,1)
    u⃗  = sort(M, dims=1, rev=true)
    λ⃗ = (1 .- cumsum(u⃗, dims=1)) ./ (1:D)
    λ⃗[u⃗ .+ λ⃗ .< 0] .= Inf
    λ = minimum(λ⃗, dims=1)
    max.(M .+ λ, 0)
end

# projects onto a smaller sub-simplex where the dimensions add to simplex_sum
function simplex_project(y⃗::AbstractVector, simplex_sum::Real)
    D = length(y⃗)
    u⃗  = sort(y⃗, rev=true)
    λ⃗ = (simplex_sum .- cumsum(u⃗)) ./ (1:D)
    λ = λ⃗[findlast(u⃗ .+ λ⃗ .> 0)]
    max.(y⃗ .+ λ, 0) .+ (1 - simplex_sum) ./ D
end

# both vectorized and sub-simplex
function simplex_project(M::AbstractMatrix, simplex_sum::Real)
    D = size(M,1)
    u⃗  = sort(M, dims=1, rev=true)
    λ⃗ = (simplex_sum .- cumsum(u⃗, dims=1)) ./ (1:D)
    λ⃗[u⃗ .+ λ⃗ .< 0] .= Inf
    λ = minimum(λ⃗, dims=1)
    max.(M .+ λ, 0) .+ (1 - simplex_sum) ./ D
end

# Normalize a distribution by flooring at zero and then dividing by the sum.
function simplex_normalize(y⃗)
    y⃗ = y⃗ .- min.(minimum(y⃗, dims=1), 0)
    y⃗ ./ sum(y⃗, dims=1)
end

# Filter a collection of mixed strategies to remove approximate duplicates.
# Will keep the first appearance of a group, so it's a good idea to pre-sort by priority.
function filter_unique(mixtures::Matrix; max_diff=1e-2)
    if size(mixtures,2) == 0
        return mixtures
    end
    unique = Matrix{eltype(mixtures)}(undef, size(mixtures)...)
    unique[:,1] .= mixtures[:,1]
    count = 1
    for m in 2:size(mixtures,2)
        is_unique = true
        for i in 1:count
            if maximum(unique[:,i] .- mixtures[:,m]) < max_diff
                is_unique = false
                continue
            end
        end
        if is_unique
            count += 1
            unique[:,count] .= mixtures[:,m]
        end
    end
    return Array(unique[:,1:count])
end

function num_payoffs(num_players::Integer, num_actions::Integer; dev=true)
    if dev
        return exp(logmultinomial(num_players-1, num_actions-1)) * num_actions
    else
        return exp(logmultinomial(num_players, num_actions-1)) * num_actions
    end
end