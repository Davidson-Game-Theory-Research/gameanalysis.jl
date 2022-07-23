using CSV, DataFrames, BenchmarkTools
# using CUDA
using Statistics: std

include("GameAnalysis.jl")

const NUM_FUNCTIONS = 200
const independent_columns = [
    "players"=>Int[], "actions"=>Int[], "batch_size"=>Int[]
]
const dependent_columns = [
    "min"=>Float64[], "max"=>Float64[], "mean"=>Float64[],
    "median"=>Float64[], "std"=>Float64[], "trials"=>Int[]
]
const input_col_indices = 1:3

function parameter_setup(; outfile_name, min_players, max_players, min_actions, max_actions, batch_sizes)
    all_inputs = Iterators.product(min_players:max_players, min_actions:max_actions, batch_sizes)
    if !isfile(outfile_name)
        timing_data_frame = DataFrame(independent_columns..., dependent_columns...)
        CSV.write(outfile_name, [], writeheader=true, header=names(timing_data_frame))
        return all_inputs
    else
        existing_data = DataFrame(CSV.File(outfile_name))
        return setdiff(all_inputs, Tuple.(eachrow(existing_data[:,input_col_indices])))
    end
end

function batched_dev_pays(game, mixtures, batch_size)
    num_mixtures = size(mixtures,2)
    dev_pays = Array{Float64}(undef, size(mixtures)...)
    for start_index in 1:batch_size:num_mixtures
        end_index = min(start_index + batch_size - 1, num_mixtures)
        dev_pays[:,start_index:end_index] = deviation_payoffs(game, mixtures[:,start_index:end_index])
    end
    return dev_pays
end

function dev_pays_timing(players, actions, batch_size; game_type, num_mixtures, memory_available, outfile_name, outfile_lock)
    timing_inputs = Dict{String,Integer}("players"=>players, "actions"=>actions, "batch_size"=>batch_size)
    timing_results = Dict{String,Real}()
    total_memory = game_size(game_type, players, actions) * batch_size
    if total_memory > memory_available
        print("batches are too large with p=", players," & a=", actions,"\n")
        for stat in ["min","max","mean","median","std"]
            timing_results[stat] = NaN
        end
        timing_results["trials"] = 0
    else
        mixtures = Matrix{Float64}(undef, actions, num_mixtures)
        grid_points, grid_size = finest_grid(actions, num_mixtures)
        mixtures[:,1:grid_size] = mixture_grid(actions, grid_points)
        if grid_size < num_mixtures
            mixtures[:,grid_size+1:end] = random_mixtures(actions, num_mixtures - grid_size)
        end
        agg = additive_sin_game(players, actions, NUM_FUNCTIONS)
        try
            sg = to_sym_game(agg, game_type)
            if game_type == GPUArrays
                t = @benchmark CUDA.@sync batched_dev_pays($sg, $mixtures, $batch_size)
            else
                t = @benchmark batched_dev_pays($sg, $mixtures, $batch_size)
            end
            times = t.times
            timing_results["min"] = minimum(times)
            timing_results["max"] = maximum(times)
            timing_results["mean"] = mean(times)
            timing_results["median"] = median(times)
            timing_results["std"] = std(times)
            timing_results["trials"] = length(times)
        catch e
            if e isa OverflowError
                print("multinomial overflows with p=", players," & a=", actions,"\n")
                for stat in ["min","max","mean","median","std"]
                    timing_results[stat] = NaN
                end
                timing_results["trials"] = 0
            else
                throw(e)
            end
        end
    end
    timing_data_frame = DataFrame(independent_columns..., dependent_columns...)
    push!(timing_data_frame, merge(timing_inputs, timing_results))
    lock(outfile_lock)
    CSV.write(outfile_name, timing_data_frame[1:1,:], writeheader=false, append=true)
    unlock(outfile_lock)
end