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
using Flux.Data: DataLoader
using Flux: throttle
using Flux: softmax, onecold
# make a nn to study these games
using Flux, CUDA
using IterTools
CUDA.allowscalar(false)

### load all results are analysis
full_results = mapreduce(deserialize, vcat, readdir("c:/data/game2048/raw/", join=true));

# countmap(map(x->maximum(x.final_state), full_results))
#   5  => 48
#   6  => 223
#   7  => 1120
#   11 => 21269
#   10 => 106756
#   9  => 41368
#   12 => 1
#   8  => 7115

# how many are bigger than 11
good_results = filter(x->maximum(x.final_state) >= 11, full_results);
length(good_results)
mapreduce(x->x.counter, +, good_results)

bad_results = filter(x->maximum(x.final_state) <=9, full_results);
mapreduce(x->x.counter, +, bad_results)

results_for_modelling = vcat(good_results, bad_results);

function make_x_for_value_modelling(results_for_modelling)
    # preallocate the memory
    res = Array{Int8}(undef, 4, 4, mapreduce(x->x.counter, +, results_for_modelling))
    i = 1
    for result in results_for_modelling
        for state in result.states
            res[:, :, i] .= bitboard_to_array(state)
            i += 1
        end
    end
    res
end

function make_y_for_value_modelling(results_for_modelling)
    # preallocate the memory
    res = Array{Int32}(undef, mapreduce(x->x.counter, +, results_for_modelling))
    i = 1
    for result in results_for_modelling
        val = sum(2 .<< bitboard_to_array(result.final_state))
        for _ in 1:result.counter
            res[i] = val
            i += 1
        end
    end
    res
end

@time x = reshape(make_x_for_value_modelling(results_for_modelling), 4, 4, 1, :);
@time y = make_y_for_value_modelling(results_for_modelling);

value_model = Chain(
    Conv((2, 2), 1=>8, relu),
    Conv((2, 2), 8=>8, relu),
    flatten,
    Dense(32, 1)
) #|> gpu


ly = length(y)
using StatsBase
idx1 = sort(sample(1:ly, round(Int, ly * 0.50), replace=false))
# idx2 = sort(sample(setdiff(1:ly, idx1), round(Int, ly * 0.1), replace=false))

xtrain = (x[:, :, :, idx1] ./ Float32(11)) #|> cu;
ytrain = Float32.(reshape(log.(y[idx1]), 1, :)) #|> cu;

using Statistics
ytrain = (ytrain .- mean(ytrain))/std(ytrain)


testidx = sample(setdiff(1:ly, idx1), round(Int, ly * 0.10), replace=false)
xtest = x[:, :, :, testidx] ./ Float32(11) #|> cu;
ytest = Float32.(reshape(log.(y[testidx]), 1, :)) #|> cu;
ytest = (ytest .- mean(ytest))/std(ytest)

loss(x, y) = Flux.mse(value_model(x), y; agg=sum)



loss(xtest, ytest)

using StatsPlots

using DataFrames
px = value_model(xtest)
df = DataFrame(px = reshape(px, :), ytest =reshape(ytest, :))

sort!(df, :ytest)



cb = Flux.throttle(()->println(loss(xtest, ytest)), 8)

opt= ADAM()

dl = DataLoader((xtrain, ytrain); batchsize=32);

@time Flux.train!(loss, Flux.params(value_model), dl, opt; cb)


