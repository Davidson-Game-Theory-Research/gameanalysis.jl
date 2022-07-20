using CUDA, Distributed
using CSV, DataFrames

global GPUs = collect(CUDA.devices())
global num_resources = length(GPUs)
addprocs(num_resources, exeflags="--project=$(Base.active_project())")
global worker_procs = workers()
@everywhere using CUDA

global resources_lock = ReentrantLock()
global resource_availablility = ones(Bool, num_resources)

@everywhere include("largest_error_experiment.jl")

@everywhere const OUTFILE = "GPUArrays_errors.csv"
const SETUP_FUNCTION = parameter_setup
const SETUP_CONFIG = Dict(
    :outfile_name=>OUTFILE,
    :min_players=>2,
    :max_players=>512,
    :min_strats=>2,
    :max_strats=>20
)
@everywhere const EXPERIMENT_FUNCTION = expected_error
@everywhere const EXPERIMENT_CONFIG = Dict(
    :game_type=>GPUArrays,
    :game_bits=>32,
    :outfile_name=>OUTFILE,
    :outfile_lock=>ReentrantLock()
)

function claim_resources()
    lock(resources_lock)
    for i in 1:num_resources
        if resource_availablility[i]
            resource_availablility[i] = false
            unlock(resources_lock)
            return i
        end
    end
    unlock(resources_lock)
    error("no available resources")
end

function release_resources(resurce_num)
    lock(resources_lock)
    resource_availablility[resurce_num] = true
    unlock(resources_lock)
end

function run_experiment(args_tuple)
    my_resource_num = claim_resources()
    my_GPU = GPUs[my_resource_num]
    my_proc = worker_procs[my_resource_num]
    remotecall_wait(my_proc) do
        device!(my_GPU)
        println("Process ", getpid(), " using ", device(), " to test ", args_tuple, "\n")
        EXPERIMENT_FUNCTION(args_tuple...; EXPERIMENT_CONFIG...)
    end
    release_resources(my_resource_num)
end

function main()
    inputs = SETUP_FUNCTION(; SETUP_CONFIG...)
    asyncmap(run_experiment, inputs; ntasks=num_resources)
end

main()