using DataFrames, CSV

function check_setup(; outfile_name, min_players, max_players, min_actions, max_actions, batch_sizes, input_col_indices)
    all_inputs = Iterators.product(min_players:max_players, min_actions:max_actions, batch_sizes)
    if !isfile(outfile_name)
    	existing_data = []
    else
        existing_data =  Tuple.(eachrow(DataFrame(CSV.File(outfile_name))[:,input_col_indices]))
    end
    remaining_inputs = setdiff(all_inputs, existing_data)
    println("Experiment summary:")
    println(length(all_inputs), " total tests")
    println(length(remaining_inputs), " tests completed.")
    println(length(all_inputs) - length(remaining_inputs), " tests remaining.")
end

const SETUP_CONFIG = Dict(
    :outfile_name=>"data/PayoffArrays_timing.csv",
    :input_col_indices=>1:3,
    :min_players=>2,
    :max_players=>128,
    :min_actions=>4,
    :max_actions=>6,
    :batch_sizes=>[1]#, 10, 100]
)

check_setup(; SETUP_CONFIG...)

