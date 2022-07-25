using CUDA, Distributed
using CSV, DataFrames

global GPUs = collect(CUDA.devices())
global num_resources = length(GPUs)
addprocs(num_resources, exeflags="--project=$(Base.active_project())")
global worker_procs = workers()
@everywhere using CUDA

global resources_lock = ReentrantLock()
global resource_availablility = ones(Bool, num_resources)

@everywhere include("dev_pays_timing_experiment.jl")

@everywhere const OUTFILE = "data/GPUArrays_timing.csv"
const SETUP_FUNCTION = parameter_setup
const SETUP_CONFIG = Dict(
    :outfile_name=>OUTFILE,
    :min_players=>2,
    :max_players=>128,
    :min_actions=>7,
    :max_actions=>8,
    :batch_sizes=>[1, 10, 100]
)
@everywhere const EXPERIMENT_FUNCTION = dev_pays_timing
@everywhere const EXPERIMENT_CONFIG = Dict(
    :game_type=>LogProbabilities,
    :num_mixtures=>1000,
    :memory_available=>2^30,
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