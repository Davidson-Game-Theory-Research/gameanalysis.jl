using Base.Threads: @spawn, threadid, nthreads

include("dev_pays_timing_experiment.jl")

const OUTFILE = "data/PayoffDict_timing.csv"
const SETUP_FUNCTION = parameter_setup
const SETUP_CONFIG = Dict(
    :outfile_name=>OUTFILE,
    :min_players=>2,
    :max_players=>128,
    :min_actions=>3,
    :max_actions=>8,
    :batch_sizes=>[1, 10, 100]
)
const EXPERIMENT_FUNCTION = dev_pays_timing
const EXPERIMENT_CONFIG = Dict(
    :game_type=>PayoffDict,
    :num_mixtures=>1000,
    :memory_available=>2^30,
    :outfile_name=>OUTFILE,
    :outfile_lock=>ReentrantLock()
)


function run_experiment(args_tuple)
    println("Thread ", threadid(), " testing ", args_tuple,)
    EXPERIMENT_FUNCTION(args_tuple...; EXPERIMENT_CONFIG...)
end

function main()
    inputs = SETUP_FUNCTION(; SETUP_CONFIG...)
    println("Initiating ", length(inputs), " tests")
    println("on config: ", EXPERIMENT_CONFIG)
    println("using ", nthreads(), " threads.")
    @sync for args in inputs
        @spawn run_experiment(args)
    end
end

main()