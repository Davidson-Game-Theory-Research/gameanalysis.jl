using CSV, DataFrames

include("GameAnalysis.jl")

const MAX_SIZE = 2^30
const NUM_MIXTURES = 1000
const NUM_GAMES = 10
const NUM_FUNCTIONS = 20

const ERROR_PERCENTILES = [.99, .999, .9999, .99999]

function tables_size(players, strategies, bits)
    return exp(logmultinomial(players-1, strategies-1)) * strategies * bits / 4
end

function parameter_setup(; outfile_name, min_players, max_players, min_strats, max_strats)
    if !isfile(outfile_name)
        error_data_frame = DataFrame("players"=>Int[], "strategies"=>Int[],
                            [string(p)*" error"=>Float64[] for p in ERROR_PERCENTILES]...)
        CSV.write(outfile_name, [], writeheader=true, header=names(error_data_frame))
    end
    return Iterators.product(min_players:max_players, min_strats:max_strats)
end

function expected_error(players, strategies; game_type, game_bits, outfile_name, outfile_lock)
    lock(outfile_lock)
    existing_data = DataFrame(CSV.File(outfile_name))
    unlock(outfile_lock)
    known_values = existing_data[existing_data.players .== players .&&
                                existing_data.strategies .== strategies, end]
    if length(known_values) > 0
        print("found existing result for p=",players," & s=",strategies,"\n")
        return
    end
    game_size = tables_size(players, strategies, game_bits)
    if game_size > MAX_SIZE
        print("game is larger than ", MAX_SIZE, "B with p=", players," & s=", strategies,"\n")
        error_quantiles = Array{Float64}(undef, size(ERROR_PERCENTILES))
        fill!(error_quantiles, NaN)
    else
        errors = Vector{Float64}()
        batch_size = floor(Int, MAX_SIZE / game_size)
        mixtures = Matrix{Float64}(undef, strategies, NUM_MIXTURES)
        grid_points, grid_size = finest_grid(actions, NUM_MIXTURES)
        mixtures[:,1:grid_size] = mixture_grid(actions, grid_points)
        if grid_size < num_mixtures
            mixtures[:,grid_size+1:end] = random_mixtures(actions, num_mixtures - grid_size)
        end
        for g in 1:NUM_GAMES
            agg = additive_sin_game(players, strategies, NUM_FUNCTIONS)
            sg = to_sym_game(agg, game_type)
            correct_dev_pays = deviation_payoffs(agg, mixtures)
            range = maximum(correct_dev_pays) - minimum(correct_dev_pays)
            for start_index in 1:batch_size:NUM_MIXTURES
                end_index = min(start_index + batch_size - 1, NUM_MIXTURES)
                dev_pays = denormalize(sg, deviation_payoffs(sg, mixtures[:,start_index:end_index]))
                append!(errors, abs.(dev_pays .- correct_dev_pays[:,start_index:end_index]) ./ range)
            end
        end
        error_quantiles = [partialsort(errors, floor(Int, p * length(errors))) for p in ERROR_PERCENTILES]
    end
    error_data_frame = DataFrame("players"=>Int[], "strategies"=>Int[],
                        [string(p)*" error"=>Float64[] for p in ERROR_PERCENTILES]...)
    push!(error_data_frame, (players, strategies, error_quantiles...))
    lock(outfile_lock)
    CSV.write(outfile_name, error_data_frame[1:1,:], writeheader=false, append=true)
    unlock(outfile_lock)
end