import CUDA
CUDA.allowscalar(false)

include("experimental_game_types/LogMultinomial.jl")
using .LogMultinomial

include("experimental_game_types/AbstractGames.jl")
include("experimental_game_types/generate_games.jl")
# include("experimental_game_types/SymCPU.jl")
# include("experimental_game_types/SymGPU.jl")
# include("experimental_game_types/LogSymCPU.jl")
include("experimental_game_types/LogSymGPU.jl")
# include("experimental_game_types/RepWeightsCPU.jl")
# include("experimental_game_types/RepWeightsGPU.jl")

include("mixed_nash.jl")
