using CSV, DataFrames, BenchmarkTools
using CUDA
using Statistics: std

include("GameAnalysis.jl")

const NUM_FUNCTIONS = 200
const independent_columns = [
    "players"=>Int[], "actions"=>Int[], "batch_size"=>Int[]
]
const dependent_columns = [
    "min_time"=>Float64[]
]
const input_col_indices = 1:3

function parameter_setup(; outfile_name, player_counts, action_counts, batch_sizes)
    all_inputs = Iterators.product(player_counts, action_counts, batch_sizes)
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

function dev_pays_timing(players, actions, batch_size; game_type, num_mixtures,
                        benchmark_evals, benchmark_samples, benchmark_seconds,
                        memory_available, outfile_name, outfile_lock)
    timing_inputs = Dict{String,Integer}("players"=>players, "actions"=>actions, "batch_size"=>batch_size)
    timing_results = Dict{String,Real}()
    total_memory = game_size(game_type, players, actions) * batch_size
    if total_memory > memory_available
        print("batches are too large with p=", players,", a=", actions, ", b=", batch_size, "\n")
        timing_results["min_time"] = NaN
    else
        mixtures = random_mixtures(actions, num_mixtures)
        agg = additive_sin_game(players, actions, NUM_FUNCTIONS)
        try
            sg = to_sym_game(agg, game_type)
            if game_type == GPUArrays
                bm = @benchmark (CUDA.@sync batched_dev_pays($sg, $mixtures, $batch_size)) evals=benchmark_evals samples=benchmark_samples seconds=benchmark_seconds
                t = minimum(bm.times) / 10^9
            else
                t = @belapsed batched_dev_pays($sg, $mixtures, $batch_size) evals=benchmark_evals samples=benchmark_samples seconds=benchmark_seconds
            end
            timing_results["min_time"] = t
        catch e
            if e isa OverflowError || e isa InexactError
                print("multinomial overflows with p=", players," & a=", actions,"\n")
                timing_results["min_time"] = NaN
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