include("../nash_timing_experiment.jl")

OUTFILE = "data/GPUArrays_GD_timing.csv"
SETUP_FUNCTION = parameter_setup
SETUP_CONFIG = Dict(
    :outfile_name=>OUTFILE,
    :players=>[2^i for i in 1:10],
    :actions=>[2,4,6,8]
)
EXPERIMENT_FUNCTION = nash_timing
EXPERIMENT_CONFIG = Dict(
    :game_type=>GPUArrays,
    :nash_alg=>gain_descent,
    :starting_points=>100,
    :memory_available=>2^30,
    :outfile_name=>OUTFILE,
    :outfile_lock=>ReentrantLock()
)