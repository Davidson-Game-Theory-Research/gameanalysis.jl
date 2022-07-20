using ThreadsX
using CSV, DataFrames

include("largest_error_experiment.jl")

const OUTFILE = "LogProbabilities_errors.csv"
const SETUP_FUNCTION = parameter_setup
const SETUP_CONFIG = Dict(
    :outfile_name=>OUTFILE,
    :min_players=>2,
    :max_players=>512,
    :min_strats=>2,
    :max_strats=>20
)
const EXPERIMENT_FUNCTION = expected_error
const EXPERIMENT_CONFIG = Dict(
    :game_type=>LogProbabilities,
    :game_bits=>64,
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