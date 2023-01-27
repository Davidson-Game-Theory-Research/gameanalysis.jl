using Base.Threads: @spawn, threadid, nthreads
using Dates

# include("experiment_params/PayoffDict_dev_pays_timing_params.jl")
# include("experiment_params/PayoffArrays_dev_pays_timing_params.jl")
# include("experiment_params/RepeatsTable_dev_pays_timing_params.jl")
# include("experiment_params/DeviationProfiles_dev_pays_timing_params.jl")
# include("experiment_params/WeightedPayoffs_dev_pays_timing_params.jl")
include("experiment_params/LogProbabilities_dev_pays_timing_params.jl")

# include("experiment_params/PayoffDict_RD_timing_params.jl")


function run_experiment(args_tuple)
    println("Thread ", threadid(), " testing input ", args_tuple, " starting at ", Dates.format(now(), "HH:MM"))
    flush(stdout)
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
