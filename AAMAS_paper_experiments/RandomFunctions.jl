using StatsBase: sample, Weights
using Distributions: Beta, Uniform, LKJ
using LinearAlgebra: I, det, inv

#=
degr_probs: probability of a polynomial with each degree, starting from 0.
min, max_coefs: arrays with the same size as degr_probs that specify the min and max
            coefficient for each term (in order starting from x^0). Coefficients are
            chosen from a beta distribution scaled to the [min,max] interval.
beta_params: parameters to pass to Distributions.Beta for generating coefficients. Can be
            either a 1D array with 2 elements: alpha and beta, or a 2D array with size
            (max_deg+1,2) with alpha and beta for each coefficient.
=#
function random_polynomial(; degr_probs=[0;.6;.4], min_coefs=[-5;-1;-.2],
                           max_coefs=[5;1;.2], beta_params=[1 1; 1 1; 2 2])
    degree = sample(0:length(degr_probs)-1, Weights(degr_probs))
    if ndims(beta_params) == 1
        coefficients = rand(Beta(beta_params...), degree+1)
    else
        coefficients = [rand(Beta(beta_params[i,:]...)) for i in 1:degree+1]
    end
    coefficients .*= max_coefs[1:degree+1,:] .- min_coefs[1:degree+1,:]
    coefficients .+= min_coefs[1:degree+1,:]
    return f(x) = coefficients' * [x.^i for i in 0:degree]
end

#=
Both period and amplitude are drawn from beta distributions (defaulting to uniform)
that are scaled to the given [min,max] range.
The phase-shift is always drawn uniformly from [0,period].
=#
function random_sin_func(; period_range=[1;10], amplitude_range=[1;10],
                         period_beta_params=[1;1], amplitude_beta_params=[1;1])
    period = rand(Beta(period_beta_params...))
    period *= period_range[2] - period_range[1]
    period += period_range[1]
    amplitude = rand(Beta(amplitude_beta_params...))
    amplitude *= amplitude_range[2] - amplitude_range[1]
    amplitude += amplitude_range[1]
    shift = rand(Uniform(0, period))
    return f(x) = amplitude .* sin.(2π .* (x .+ shift) ./ period)
end


function random_gaussian(mean_distr; mean_scale=1, var_scale=1, covar_scale=1, corr_strength=1)
    μ⃗ = rand(mean_distr) .* mean_scale
    d = length(μ⃗)
    Σ = rand(LKJ(d, corr_strength))
    Σ[I(d)] .*= var_scale
    Σ[.!I(d)] .*= covar_scale
    inverse = inv(Σ)
    determinant = det(Σ)
    return N(s⃗) = exp( -(s⃗ - μ⃗)' * inverse * (s⃗ - μ⃗) / 2 ) / √( (2π)^d * determinant )
end