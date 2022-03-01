# I want to try a pure MC simulation that doesn't rely on a neural network and
# see how things
using Statistics: mean, quantile
using StatsBase: countmap
using Game2048
using Game2048: bitboard_to_array, DIRS, transform_4corners_to_4cols
using Serialization: serialize, deserialize
using Base.Threads
using BSON
using BenchmarkTools
using Flux
using Flux.Data: DataLoader
using Flux: throttle, softmax, onecold, onehot, logitbinarycrossentropy, onehotbatch
# make a nn to study these games
using CUDA
using Chain
CUDA.allowscalar(false)

# this is the neural network that will be trained
model_group_cnn = Flux.Chain(
    Conv((4,1), 1=>64, relu),
    x->reshape(x, 2, 2, 64, :),
    Conv((2,2), 64=>64),
    Flux.flatten,
    SkipConnection(Dense(64, 64, relu), (mx,x) -> mx.+x),
    SkipConnection(Dense(64, 64), (mx,x) -> mx.+x),
    Dense(64, 4)
) |> gpu

model_group_cnn(rand(Float32, 4, 4, 1, 100) |> cu);

# the policy which take a bitboard
function group_cnn_policy(bitboard::Bitboard)
    @chain bitboard begin
        bitboard_to_array
        transform_4corners_to_4cols
        reshape(_, 4, 4, 1, :) ./ Float32(11)
        cu
        model_group_cnn
        softmax
        cpu
        reshape(4)
    end
end

function prep_bitboard_for_nn(bitboard::Bitboard)
    @chain bitboard begin
        bitboard_to_array
        transform_4corners_to_4cols
        reshape(_, 4, 4, 1, :) ./ Float32(11)
        # cu
    end
end

bb = initbboard()

prep_bitboard_for_nn(bb)

bb
rotl90(bb)

mod1.((1:4).-1, 4)

# this is the matrix that rotates outputs by 90 degrees
rotl90m = cu(Float32.([0 0 0 1; 1 0 0 0; 0 1 0 0; 0 0 1 0]))

x = model_group_cnn(prep_bitboard_for_nn(bb))
x1 = model_group_cnn(prep_bitboard_for_nn(rotl90(bb)))

cu(Float32.(reshape(1:16, 4, 4)))

model_group_cnn(prep_bitboard_for_nn(bb |> rotl90 |> rotl90))
model_group_cnn(prep_bitboard_for_nn(bb |> rotl90 |> rotl90 |> rotl90))

x = prep_bitboard_for_nn(bb)
y = cu(Float32.([0, 0, 0, 1]))

x_l90 = prep_bitboard_for_nn(rotl90(bb))
x_l180 = prep_bitboard_for_nn(rotl90(bb) |> rotl90)
x_l270 = prep_bitboard_for_nn(rotl90(bb) |> rotl90 |> rotl90)


const mean_matrix = cu(fill(0.25, 4, 4))

# this loss will try and make the loss produced by the network to be consistent
function different_dir_loss(x)
    reshape_x = reshape(x, 4, 4, :)
    model_group_cnn(reshape(batched_mul(reshape_x, rotl90m), 4, 4, 1, :))

    # x_l90 = reshape(reshape_x * rotl90m, 4, 4, 1, :)
    # x_l180 = reshape(reshape_x * rotl90m * rotl90m, 4, 4, 1, :)
    # x_l270 = reshape(reshape_x * rotl90m * rotl90m * rotl90m, 4, 4, 1, :)
    # cnns = cat(
    #     model_group_cnn(x),
    #     rotl90m*model_group_cnn(x_l90),
    #     rotl90m*rotl90m*model_group_cnn(x_l180),
    #     rotl90m*rotl90m*rotl90m*model_group_cnn(x_l270),
    #     dims=2)

    # sum((cnns .- cnns*mean_matrix).^2)
end

# different_dir_loss(x)

opt=ADAM()

Flux.@epochs 100 Flux.train!(different_dir_loss, Flux.params(model_group_cnn), [(x,)], opt)


logitbinarycrossentropy(model_group_cnn(x), y)
logitbinarycrossentropy(rotl90m*model_group_cnn(x_l90), y)
logitbinarycrossentropy(rotl90m*rotl90m*model_group_cnn(x_l90), y)

# load the data

@time games = deserialize.(readdir(raw"C:\data\game2048\raw-1000"; join=true));

games = filter(game->maximum(game.final_state) >= 11, reduce(vcat, games))

# take 1 gamerecorder to matrices
all_game_states = mapreduce(game->game.states, vcat, games)

game_states = mapreduce(prep_bitboard_for_nn, (x,y)->cat(x, y, dims=4), all_game_states) |> cpu

CUDA.@time different_dir_loss(game_states);

# this game states tensor can be feed into the network

moves = Flux.onehotbatch(mapreduce(game->game.moves, vcat, games), Game2048.DIRS)

logitbinarycrossentropy(model_group_cnn(game_states |> cu), moves |> cu)

opt=ADAM()


dl = DataLoader((game_states |> cu, moves |> cu); batchsize=32, shuffle=true);

function loss(x,y)
    logitbinarycrossentropy(model_group_cnn(x), y)
end

cb() = begin

end

Flux.train!(loss, Flux.params(model_group_cnn), dl, opt)



function train_n_times(games)
    model_group_cnn_cpu = model_group_cnn |> cpu

    function group_cnn_policy(bitboard)
        @chain bitboard begin
            bitboard_to_array
            transform_4corners_to_4cols
            reshape(_, 4, 4, 1, :) ./ Float32(11)
            model_group_cnn_cpu
            softmax
            reshape(4)
        end
    end
    ## which one is it predicting better
    game1 = Game2048.play_game_with_policy_w_record(group_cnn_policy, initbboard());

    # For each state compute their policy
    policies = reduce((x,y)->cat(x,y, dims=2), group_cnn_policy.(game1.states))

    # progress move backwards until we can't reach 11
    goback_i = go_back(game1)

    # assess each state by finding the best move
    best_moves_according_to_random = Game2048.find_best_move.(@view(game1.states[1:end-goback_i]), 100, randompolicy)
    # countmap(best_moves_according_to_random)

    push!(games, (game1, best_moves_according_to_random))

    lossval = logitbinarycrossentropy(policies[:,1:end-goback_i], onehotbatch(Int.(best_moves_according_to_random) .+ 1, 1:4))
    acc = mean(onecold(policies[:,1:end-goback_i]) .== Int.(best_moves_according_to_random) .+ 1)
    println("accuracy: $(acc); loss $(lossval); best: $(maximum(game1.final_state))")

    # find the move that it's the surest about
    most_sure_move = argmax(policies)[2]
    println("")
    display(game1.states[most_sure_move])
    println("move made: $(game1.moves[most_sure_move])")
    println("policy $(policies[:, most_sure_move])")
    println("maxpolicy $(maximum(policies[:, most_sure_move]))")
    println("best policy move: $(DIRS[argmax(policies[:, most_sure_move])])")
    println("best move: $(best_moves_according_to_random[most_sure_move])")

    # trim out that many moves and make it learn bitches
    new_train_x = reduce((x,y)->cat(x, y, dims=4), make_new_training_data.(@view(game1.states[1:end-goback_i]))) |> cu
    new_train_y = onehotbatch(Int.(best_moves_according_to_random) .+ 1, 1:4) |> cu


    Flux.train!(loss, Flux.params(model_group_cnn), [(new_train_x, new_train_y)], opt);
end