# I want to try a pure MC simulation that doesn't rely on a neural network and
# see how things go
using Statistics: mean, quantile
using StatsBase: countmap, Weights, sample
using Game2048, Game2048Core
using Game2048: play_game_with_policy
using Game2048Core: bitboard_to_array, DIRS, count0, value
using Serialization: serialize, deserialize
using Base.Threads
using BSON
using BenchmarkTools
using Game2048: inittree, mcts!, maxdepth, displaytop, play_game_via_mcts_w_tries


# includet("src/mcts-utils.jl")
# includet("src/neural-network-utils.jl")

if false
    # this is loading the model
    # model_cpu = BSON.load("model_cpu")[:model_cpu]
    # model_cpu = BSON.load("model_cpu_rotational_cnn")[:model_cpu]

    model_cpu(Float32.(bitboard_to_array(initbboard())))

    function bitboard_arr_to_rotational_format(bboard)
        model_cpu(Float32.(transform_4corners_to_4cols(bboard)))
    end

    bitboard_arr_to_rotational_format(bitboard_to_array(initbboard()))

    cnnpolicy_cpu = nn_policy_maker(bitboard_arr_to_rotational_format)
end

tree = inittree(randompolicy)
tree.state

@time mcts!(tree, 888)

mcts!(tree)
maxdepth(tree)

tree


@time displaytop(mcts!(tree, 2888))

# takes about 30 minutes
# using Alert
# @time play_game_via_mcts(randompolicy; ms=100, verbose=true)
# @time  x = [play_game_via_mcts(randompolicy; ms=100) for _ in 1:100];
# countmap(map(maximum, x))
@time x1 = [play_game_via_mcts_w_timer(randompolicy; ms = 10) for _ in 1:300];
countmap(map(maximum, x1))
alert()

@time res = [play_game_via_mcts_w_tries(randompolicy; n = 600) for _ in 1:10]


using DataFrames, DataFrameMacros, Chain
using StatsBase

a = DataFrame(score = score.(res), maxtile = maximum.(res))

@chain a begin
    @combine(mean(:score), length(:score))
end

countmap(a.maxtile)

tree = inittree(randompolicy)
mcts!(tree, 10)
maxdepth(tree)

res[1]


function ok()
    i = 0
    while true
        final_state = play_game_via_mcts()
        i += 1
        if maximum(final_state) == 11
            println(i)
            return final_state
        end
    end
    println(final_state)
end

@time ok()
