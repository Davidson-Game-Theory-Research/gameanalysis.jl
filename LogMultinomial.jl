module Multinomials

export CwR, multinomial, lmultinomial, lfactorial

import Combinatorics: with_replacement_combinations, multinomial
import SpecialFunctions: lfactorial

const CwR = with_replacement_combinations

function lmultinomial(k...)
    numerator = 0
    denominator = 0.0
    @inbounds for i in k
        numerator += i
        denominator += lfactorial(i)
    end
    return lfactorial(numerator) - denominator
end

end
