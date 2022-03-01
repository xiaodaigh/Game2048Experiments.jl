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
using Flux: onehot, logitbinarycrossentropy, onehotbatch
CUDA.allowscalar(false)

@time x = deserialize("x-data.jls");
@time y = deserialize("y-data.jls");

sample_size = round(Int, size(x, 4)*0.99)

train_x = x[:, :, :, 1:sample_size] |> cu;
test_x = x[:, :, :, sample_size+1:end] |> cu;

train_y = y[:, 1:sample_size] |> cu;
test_y = y[:, sample_size+1:end] |> cu;

x=nothing; y=nothing;GC.gc() # free up memory

loss(x, y) = logitbinarycrossentropy(model(x), y)

# model = nothing
# if isfile("model_cpu")
#     model = BSON.load("model_cpu")[:model_cpu] |> gpu
# else
    model = Chain(
        Conv((2,2), 1=>16, relu),
        #Conv((2,2), 16=>16, relu),
        Conv((2,2), 16=>16),
        flatten,
        Dense(64, 4)
    ) |> gpu
# end

@time softmax(model(test_x))
@time softmax(model(test_x))
@time loss(test_x, test_y)
@time loss(test_x, test_y)



opt=ADAM()

dl = DataLoader((train_x, train_y); batchsize=2^19, shuffle=true);

accuracy(x, y) = mean(onecold(softmax(model(x))) .== onecold(y))

# @time accuracy(train_x, train_y)


last_loss = loss(test_x, test_y)
last_acc = accuracy(test_x, test_y)
worse_cnt = 0

function evalcb()
    global last_loss, worse_cnt, last_acc
    new_loss = loss(test_x, test_y)
    new_acc = accuracy(test_x, test_y)
    # println("train loss: $(loss(train_x, train_y)) train accuracy: $(accuracy(train_x, train_y))")
    println("test loss: $(new_loss) test accuracy: $(new_acc)")

    if (new_loss < last_loss) | (new_acc > last_acc)
        last_loss = new_loss
        last_acc = new_acc
        worse_cnt = 0
    else
        worse_cnt += 1
        if worse_cnt == 8
            Flux.stop()
        end
    end
end

@time evalcb()

cb = throttle(evalcb, 8)

dl2 = DataLoader((train_x[:,:,:,1:100], train_y[:, 1:100]); batchsize=32, shuffle=true);

@time Flux.train!(loss, Flux.params(model), dl2, opt; cb); # compile
@time Flux.train!(loss, Flux.params(model), dl, opt; cb); # compile
@time Flux.@epochs 8 Flux.train!(loss, Flux.params(model), dl, opt; cb);
@time Flux.@epochs 88 Flux.train!(loss, Flux.params(model), dl, opt; cb);
@time Flux.@epochs 888 Flux.train!(loss, Flux.params(model), dl, opt; cb);
@time Flux.@epochs 8888 Flux.train!(loss, Flux.params(model), dl, opt; cb);

model_cpu = model |> cpu
# @time BSON.@save "model_cpu" model_cpu

pcpu = Flux.params(model_cpu)
serialize("pcpu", pcpu)
pcpu = deserialize("pcpu")
Flux.loadparams!(model_cpu, pcpu)

# I want to see which move it's the most sure about

model(test_x)

mtx = softmax(model(test_x)) |> cpu

full_results = mapreduce(deserialize, vcat, readdir("c:/data/game2048/raw/", join=true)[2:2]);

one_game = full_results[1]

# which move did I regret the most?

game_matrix = cu(mapreduce(b->reshape(bitboard_to_array(b), 4, 4, 1, :), (a,b)->cat(a,b, dims=4), one_game.states) ./ 11)

moves_idx = Int.(one_game.moves) .+ 1

regret = softmax(model(game_matrix)) |> cpu

# find the move that regret most
_, pos = findmax([maximum(r)/r[idx] for (r, idx) in zip(eachslice(regret, dims=2), moves_idx)])


one_game.states[pos]
one_game.moves[pos] # the move that was made

# best move according to random play
countmap([Game2048.find_best_move(one_game.states[pos], 1000, Game2048.randompolicy) for i in 1:100])

prob = one_game.states[pos] |>
    bitboard_to_array |>
    (x->reshape(x, 4, 4, 1, :) ./ 11) |>
    cu |> model |> softmax

best_move = prob |> onecold |> cpu |>
    (x->DIRS[x[1]])





# the rest is just checking to see how the recommendations change in relations to recommendations
one_game.states[pos] |>
    bitboard_to_array |>
    (x->reshape(x, 4, 4, 1, :) ./ 11) |>
    cu |> model |> softmax

one_game.states[pos] |>
    bitboard_to_array |>
    rotl90 |>
    (x->reshape(x, 4, 4, 1, :) ./ 11) |>
    cu |> model |> softmax

one_game.states[pos] |>
    bitboard_to_array |>
    rotr90 |>
    (x->reshape(x, 4, 4, 1, :) ./ 11) |>
    cu |> model |> softmax

one_game.states[pos] |>
    bitboard_to_array |>
    rot180 |>
    (x->reshape(x, 4, 4, 1, :) ./ 11) |>
    cu |> model |> softmax