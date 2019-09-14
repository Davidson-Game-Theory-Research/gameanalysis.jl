module LogMultinomial

export lmultinomial

import SpecialFunctions: lfactorial

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
