include("../nash_timing_experiment.jl")

const OUTFILE = "data/PayoffDict_RD_timing.csv"
const SETUP_FUNCTION = parameter_setup
const SETUP_CONFIG = Dict(
    :outfile_name=>OUTFILE,
    :players=>[2^i for i in 1:10],
    :actions=>[2,4,6,8]
)
const EXPERIMENT_FUNCTION = nash_timing
const EXPERIMENT_CONFIG = Dict(
    :game_type=>PayoffDict,
    :nash_alg=>replicator_dynamics,
    :starting_points=>100,
    :memory_available=>2^30,
    :outfile_name=>OUTFILE,
    :outfile_lock=>ReentrantLock()
)