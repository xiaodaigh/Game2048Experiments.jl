# I want to try a pure MC simulation that doesn't rely on a neural network and
# see how things
using Statistics: mean, quantile
using StatsBase: countmap
using Game2048
using Game2048: bitboard_to_array, DIRS
using Serialization: serialize, deserialize
using Base.Threads
using BSON
using BenchmarkTools


### load all results are analysis
full_results = mapreduce(deserialize, vcat, readdir("c:/data/game2048/raw/", join=true));

ls = length.(full_results)

quantile(ls, 0.90)

normalize_bitboard(bb) = Float32.(bb ./ 11)

function make_matrix_x(grvec::Vector{GameRecorder})
    res = zeros(Float32, 4, 4, sum(length.(grvec)))
    i = 1
    for gr in grvec
        for state in gr.states
            @inbounds res[:, :, i] .= normalize_bitboard(bitboard_to_array(state))
            i += 1
        end
    end
    reshape(res, 4, 4, 1, :)
end

function make_matrix_y(grvec::Vector{GameRecorder})
    moves = mapreduce(gr->gr.moves, vcat, grvec)
    onehotbatch(moves, DIRS)
end

# filter out the best results
good_results = filter(x->x.final_state |> maximum >= 11,  full_results);

length(good_results)

@time x = make_matrix_x(good_results);
@time y = make_matrix_y(good_results);

serialize("x-data.jls", x)
serialize("y-data.jls", y)
