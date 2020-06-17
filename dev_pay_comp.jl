include("to_include.jl")
import Distributions: Normal
import Base: GC

# function dev_pay_test_games(P, A)
#     sym_CPU = SymGame_CPU(P, A, gen_test_payoffs)
#     sym_GPU = SymGame_GPU(P, A, gen_test_payoffs)
#     log_CPU = LogSymGame_CPU(P, A, gen_test_payoffs)
#     log_GPU = LogSymGame_GPU(P, A, gen_test_payoffs)
#     return sym_CPU, sym_GPU, log_CPU, log_GPU
# end

player_range = 5:5:60
action_range = 2:6

function check_dev_pays(type, players, actions, rtol=1e-6)
    game = type(players, actions, gen_test_payoffs)
    prof = uniform_mixture(game)
    pays = deviation_payoffs(game, prof)
    expected = gen_test_payoffs(zeros(1,actions))
    #println(type, " ", players, "p", actions, "a: ", isapprox(pays, expected))
    isapprox(pays, expected, rtol=rtol)
end

function player_threshold_binary_search(type, actions, rtol=1e-6, verbose=false)
    lower_bound = 2
    upper_bound = Inf
    while lower_bound < upper_bound - 1
        if verbose
            println(lower_bound, " - ", upper_bound)
        end
        if upper_bound == Inf
            players = 2*lower_bound
        else
            players = max((lower_bound + upper_bound) ÷ 2, lower_bound + 1)
        end
        if check_dev_pays(type, players, actions, rtol)
            lower_bound = players
        else
            upper_bound = players
        end
    end
    return lower_bound
end

function player_thresholds_linear_search(type, player_range, action_range, rtol=1e-6, verbose=false)
    is_close = fill(-1, size(action_range, 1), size(player_range, 1))
    for (a,actions) in enumerate(action_range)
        for (p,players) in enumerate(player_range)
            GC.gc()
            try
                is_close[a,p] = check_dev_pays(type, players, actions, rtol)
            catch e
                if isa(e, InterruptException)
                    return is_close
                end
                println("error: ",e," on a=",actions,", p=",players)
                break
            end
            if verbose
                println(actions,", ",players,": ",is_close[a,p])
            end
        end
    end
    return is_close
end

# for (p,players) in enumerate(player_range)
#     for (a,actions) in enumerate(action_range)
#         sym_CPU, sym_GPU, log_CPU, log_GPU = dev_pay_test_games(players,actions)
#         prof = uniform_mixture(sym_CPU);
#         expected = gen_test_payoffs(zeros(1,actions))
#         println(players, "p, ", actions, "a:")
#         scdp = deviation_payoffs(sym_CPU, prof)
#         println("  sym_CPU: ", round.(scdp, digits=3), " ", isapprox(scdp,expected))
#         sgdp = deviation_payoffs(sym_GPU, prof)
#         println("  sym_GPU: ", round.(sgdp, digits=3), " ", isapprox(sgdp,expected))
#         lcdp = deviation_payoffs(log_CPU, prof)
#         println("  log_CPU: ", round.(lcdp, digits=3), " ", isapprox(lcdp,expected))
#         lgdp = deviation_payoffs(log_GPU, prof)
#         println("  log_GPU: ", round.(lgdp, digits=3), " ", isapprox(lgdp,expected))
#     end
# end
