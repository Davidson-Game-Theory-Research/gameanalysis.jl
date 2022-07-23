using ThreadsX

include("dev_pays_timing_experiment.jl")

const OUTFILE = "data/PayoffArrays_timing.csv"
const SETUP_FUNCTION = parameter_setup
const SETUP_CONFIG = Dict(
    :outfile_name=>OUTFILE,
    :min_players=>2,
    :max_players=>128,
    :min_actions=>3,
    :max_actions=>6,
    :batch_sizes=>[1, 10, 100]
)
const EXPERIMENT_FUNCTION = dev_pays_timing
const EXPERIMENT_CONFIG = Dict(
    :game_type=>PayoffArrays,
    :num_mixtures=>1000,
    :memory_available=>2^30,
    :outfile_name=>OUTFILE,
    :outfile_lock=>ReentrantLock()
)


function run_experiment(args_tuple)
    print("Thread ", Threads.threadid(), " testing ", args_tuple, "\n")
    EXPERIMENT_FUNCTION(args_tuple...; EXPERIMENT_CONFIG...)
end

function main()
    inputs = SETUP_FUNCTION(; SETUP_CONFIG...)
    ThreadsX.foreach(run_experiment, inputs)
end

main()