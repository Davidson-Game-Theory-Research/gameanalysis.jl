#=
logmultinomial computes the log of the multinomial coefficient.
From Wolfram Mathworld (https://mathworld.wolfram.com/MultinomialCoefficient.html):

The multinomial coefficients

$$(n_1, n_2, \ldots, n_k)! = \frac{(n_1+n_2+...+n_k)!}{n_1! n_2! \ldots n_k!}$$

are the terms in the multinomial series expansion. In other words, the number
of distinct permutations in a multiset of $k$ distinct elements of multiplicity
$n_i$ $(1 \le i \le k)$ is $(n_1, \ldots, n_k)!$.
=#

using SpecialFunctions: loggamma

function logmultinomial(multiset...)
    numerator = 0
    denominator = 0.0
    @inbounds for multiplicity in multiset
        numerator += multiplicity
        denominator += loggamma(multiplicity + 1)
    end
    return loggamma(numerator + 1) - denominator
end
