using TensorCast
using CUDA
CUDA.allowscalar(false)

C = cu(ones(2,10))
L = cu(ones(3,10))
# C = ones(15,3)
# L = ones(15,5)

@reduce D[m,a] := sum(p) C[a,p] + L[m,p]
# @cast D[p,m,a] := C[p,a] + L[p,m]
# D = reshape(sum(D, dims=1), (3,2))
