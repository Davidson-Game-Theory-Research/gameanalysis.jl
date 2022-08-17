@everywhere using CUDA
@everywhere using CSV, DataFrames
@everywhere using Dates

global GPUs = collect(CUDA.devices())
global num_resources = length(GPUs)
global worker_procs = workers()
global resources_lock = ReentrantLock()
global resource_availablility = ones(Bool, num_resources)

@everywhere include("experiment_params/GPUArrays_dev_pays_timing_params.jl")
# @everywhere include("experiment_params/GPUArrays_RD_timing_params.jl")
# @everywhere include("experiment_params/GPUArrays_GD_timing_params.jl")

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
        println("Process ", getpid(), " using ", device(), " to test ", args_tuple, " starting at ", Dates.format(now(), "HH:MM"))
        flush(stdout)
        EXPERIMENT_FUNCTION(args_tuple...; EXPERIMENT_CONFIG...)
    end
    release_resources(my_resource_num)
end

function main()
    inputs = SETUP_FUNCTION(; SETUP_CONFIG...)
    println("Initiating ", length(inputs), " tests")
    println("on config: ", EXPERIMENT_CONFIG)
    println("using ", num_resources, " GPUs.")
    asyncmap(run_experiment, inputs; ntasks=num_resources)
end

main()