using CSV, DataFrames, BenchmarkTools
# using CUDA
using Statistics: std

include("GameAnalysis.jl")

const NUM_FUNCTIONS = 200
const independent_columns = [
    "players"=>Int[], "actions"=>Int[]
]
const dependent_columns = [
    "min"=>Float64[], "max"=>Float64[], "mean"=>Float64[],
    "median"=>Float64[], "std"=>Float64[], "trials"=>Int[]
]
const input_col_indices = 1:2

function parameter_setup(; outfile_name, players, actions)
    all_inputs = Iterators.product(players, actions)
    if !isfile(outfile_name)
        timing_data_frame = DataFrame(independent_columns..., dependent_columns...)
        CSV.write(outfile_name, [], writeheader=true, header=names(timing_data_frame))
        return all_inputs
    else
        existing_data = DataFrame(CSV.File(outfile_name))
        return setdiff(all_inputs, Tuple.(eachrow(existing_data[:,input_col_indices])))
    end
end

function nash_timing(players, actions; game_type, nash_alg, starting_points, memory_available, outfile_name, outfile_lock)
    timing_inputs = Dict{String,Integer}("players"=>players, "actions"=>actions)
    timing_results = Dict{String,Real}()
    memory_required = game_size(game_type, players, actions)
    if memory_required > memory_available
        println("game is large with p=", players,", a=", actions)
        for stat in ["min","max","mean","median","std"]
            timing_results[stat] = NaN
        end
        timing_results["trials"] = 0
    else
        initial_mixtures = Matrix{Float64}(undef, actions, starting_points)
        grid_points, grid_size = finest_grid(actions, starting_points)
        initial_mixtures[:,1:grid_size] = mixture_grid(actions, grid_points)
        if grid_size < starting_points
            initial_mixtures[:,grid_size+1:end] = random_mixtures(actions, starting_points - grid_size)
        end
        batch_size = Int(round(min(memory_available ÷ memory_required, starting_points)))
        agg = additive_sin_game(players, actions, NUM_FUNCTIONS)
        try
            sg = to_sym_game(agg, game_type)
            if game_type == GPUArrays
                t = CUDA.@sync @benchmark batch_nash($nash_alg, $sg, $initial_mixtures, $batch_size)
            else
                t = @benchmark batch_nash($nash_alg, $sg, $initial_mixtures, $batch_size)
            end
            times = t.times
            timing_results["min"] = minimum(times)
            timing_results["max"] = maximum(times)
            timing_results["mean"] = mean(times)
            timing_results["median"] = median(times)
            timing_results["std"] = std(times)
            timing_results["trials"] = length(times)
        catch e
            if e isa OverflowError || e isa InexactError
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