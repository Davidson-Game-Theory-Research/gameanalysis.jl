include("../dev_pays_timing_experiment.jl")

OUTFILE = "data/GPUArrays_dev_pays_timing.csv"
SETUP_FUNCTION = parameter_setup
SETUP_CONFIG = Dict(
    :outfile_name=>OUTFILE,
    :player_counts=>[4,6,8,12,16,24,32,48,64,96,128,192,256,384,768,1024],
    :action_counts=>[4,6,8],
    :batch_sizes=>[1,8,64]
)
EXPERIMENT_FUNCTION = dev_pays_timing
EXPERIMENT_CONFIG = Dict(
    :game_type=>GPUArrays,
    :num_mixtures=>1024,
    :benchmark_evals=>4,
    :benchmark_samples=>256,
    :benchmark_seconds=>60,
    :memory_available=>2^30,
    :outfile_name=>OUTFILE,
    :outfile_lock=>ReentrantLock()
)