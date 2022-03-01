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
using Flux.Data: DataLoader
using Flux: throttle
using Flux: softmax, onecold
# make a nn to study these games
using Flux, CUDA
using Flux: onehot, logitbinarycrossentropy, onehotbatch
using Chain
CUDA.allowscalar(false)

model_group_cnn = Flux.Chain(
    Conv((4,1), 1=>128, relu),
    x->reshape(x, 2, 2, 128, :),
    Conv((2,2), 128=>64),
    Flux.flatten,
    SkipConnection(Dense(64, 64, relu), (mx,x) -> mx.+x),
    SkipConnection(Dense(64, 64), (mx,x) -> mx.+x),
    Dense(64, 4)
) |> gpu

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


function get_move_probs(one_game::GameRecorder)
    # which move did I regret the most?
    game_matrix = mapreduce((a,b)->cat(a,b, dims=4), one_game.states) do state
        @chain state begin
            bitboard_to_array
            _ ./ 11
            transform_4corners_to_4cols
            reshape(4, 4, 1)
        end
    end |> cu

    softmax(model_group_cnn(game_matrix)) |> cpu
end


if false && isfile("group_pcpu")
    pcpu = deserialize("group_pcpu")
    Flux.loadparams!(model_group_cpu, pcpu)
end

# includet("src/neural-network-utils.jl")
# includet("src/policy-player.jl")

@time y = deserialize("y-data.jls");

if false
    @time x = deserialize("x-data.jls");
    @time x = mapslices(transform_4corners_to_4cols, x, dims=[1,2,3]); # about 1min
    @time serialize("x-data-corner-transformed.jls", x);
end

@time x = deserialize("x-data-corner-transformed.jls");

sample_size = round(Int, size(x, 4)*0.99)

train_x = x[:, :, :, 1:sample_size] |> cu;
test_x = x[:, :, :, sample_size+1:end] |> cu;

train_y = y[:, 1:sample_size] |> cu;
test_y = y[:, sample_size+1:end] |> cu;

x=nothing; y=nothing;GC.gc() # free up memory



loss(x, y) = logitbinarycrossentropy(model_group_cnn(x), y)

@time softmax(model_group_cnn(test_x))
@time softmax(model_group_cnn(test_x))
@time loss(test_x, test_y)
@time loss(test_x, test_y)



opt=ADAM()

dl = DataLoader((train_x, train_y); batchsize=2^5, shuffle=true);

accuracy(x, y) = mean(onecold(softmax(model_group_cnn(x))) .== onecold(y))

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

cb = throttle(evalcb, 88)

dl2 = DataLoader((train_x[:,:,:,1:100], train_y[:, 1:100]); batchsize=32, shuffle=true);

@time Flux.train!(loss, Flux.params(model_group_cnn), dl2, opt; cb); # compile
@time Flux.train!(loss, Flux.params(model_group_cnn), dl, opt; cb); # compile
@time Flux.@epochs 8 Flux.train!(loss, Flux.params(model_group_cnn), dl, opt; cb);
@time Flux.@epochs 88 Flux.train!(loss, Flux.params(model_group_cnn), dl, opt; cb);
@time Flux.@epochs 888 Flux.train!(loss, Flux.params(model_group_cnn), dl, opt; cb);
# @time Flux.@epochs 8888 Flux.train!(loss, Flux.params(model_group_cnn), dl, opt; cb);

### let's play the game vs random policy without search and see if it's better than random already


# function compareonce()
#     bboard = initbboard()
#     m1 = maximum(Game2048.play_game_with_policy(randompolicy, bboard))
#     m2 =maximum(Game2048.play_game_with_policy(group_cnn_policy, bboard))
#     m1, m2
# end

# yy = [compareonce() for _ in 1:1000]

# using StatsBase
# mean([m1-m2 for (m1, m2) in yy])

# countmap([m1 for (m1, m2) in yy])
# countmap([m2 for (m1, m2) in yy])


function go_back(game1)
    i = 1
    m1 = maximum(Game2048.play_via_monte_carlo_wo_recorder(randompolicy; n=100, bitboard = game1.states[end-i]))
    while m1 < 11
        i += 1
        m1 = maximum(Game2048.play_via_monte_carlo_wo_recorder(randompolicy; n=100, bitboard = game1.states[end-i]))
    end
    return i
end

function make_new_training_data(bitboard)
    @chain bitboard begin
        bitboard_to_array
        transform_4corners_to_4cols
        reshape(_, 4, 4, 1, :) ./ Float32(11)
    end
end


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

games = []
@time train_n_times(games)

while true
    train_n_times(games)
end

model_group_cnn_cpu = model_group_cnn |> cpu
pcpu = Flux.params(model_group_cnn_cpu)
serialize("group_pcpu", pcpu)
pcpu = deserialize("group_pcpu")
Flux.loadparams!(model_group_cnn_cpu, pcpu)


@time x = [maximum(Game2048.play_via_monte_carlo_wo_recorder(randompolicy; n=10)) for _ in 1:10]
@time y = [maximum(Game2048.play_via_monte_carlo_wo_recorder(group_cnn_policy; n=30)) for _ in 1:10]

@time xx1 = [Game2048.play_via_monte_carlo_wo_recorder(randompolicy; n=1) for _ in 1:10]
@time xx2 = [Game2048.play_via_monte_carlo_wo_recorder(group_cnn_policy; n=1) for _ in 1:10]

maximum.(xx1)
maximum.(xx2)

b = initbboard()

group_cnn_policy(b)
group_cnn_policy(rotl90(b))
group_cnn_policy(rotl90(b) |> rotl90)
group_cnn_policy(rotl90(b) |> rotl90 |> rotl90)

function rotate4(board)
    reduce((x,y)->cat(x, y, dims=3), (board, rotl90(board), rot180(board), rotr90(board)))
end

board = bitboard_to_array(initbboard())
rotate4(board)

function normalize4(board)
    reshape(board[1], 4, 4, 1, :) ./ Float32(11),
    reshape(board[2], 4, 4, 1, :) ./ Float32(11),
    reshape(board[3], 4, 4, 1, :) ./ Float32(11),
    reshape(board[4], 4, 4, 1, :) ./ Float32(11)
end

cat(board, rotr90(board), dims=3)

function rotated_cnn(bitboard)
    @chain bitboard begin
        bitboard_to_array
        rotate4
        mapslices(transform_4corners_to_4cols, _, dims=(1,2))
        reshape(_, 4, 4, 1, 4) ./ Float32(11)
        model_group_cnn_cpu
    end
end

b = initbboard()


function firstpart(bitboard)
    @chain bitboard begin
        bitboard_to_array
        rotate4
    end
end

function circular!(x)
    x[1], x[2], x[3], x[4] = x[2], x[3], x[4], x[1]
    x
end

rb = rotated_cnn(b)

rb[:, 2] .= circular!(rb[:, 2])
rb[:, 3] .= circular!(circular!(rb[:, 3]))
rb[:, 4] .= circular!(circular!(circular(rb[:, 4])))

using StatsBase
mapslices(rb, dims=2) do slice
    sum((slice .- mean(slice)).^2)
end |> sum


# @time BSON.@save "model_cpu" model_cpu

pcpu = Flux.params(model_group_cpu)
serialize("group_pcpu", pcpu)
pcpu = deserialize("group_pcpu")
Flux.loadparams!(model_group_cpu, pcpu)

# I want to see which move it's the most sure about

full_results = mapreduce(deserialize, vcat, readdir("c:/data/game2048/raw/", join=true)[2:2]);



one_game = full_results[1]



move_probs1 = move_probs(one_game)
# find the move that regret most
moves_idx = Int.(one_game.moves) .+ 1


regret_scores = [maximum(r)/r[idx] for (r, idx) in zip(eachslice(move_probs1, dims=2), moves_idx)]
pos = argmax(regret_scores)
state, actual_move pos =  one_game.states[pos], one_game.moves[pos] # the move that was made

pos, state, actual_move = regrets(one_game)

state


# best move according to random play
countmap([Game2048.find_best_move(one_game.states[pos], 1000, Game2048.randompolicy) for i in 1:100])

# this approach makes the
prob = @chain one_game.states[pos] begin
    bitboard_to_array
    transform_4corners_to_4cols
    reshape(_, 4, 4, 1, :) ./ 11
    cu
    model_group_cnn
    softmax
    cpu
    reshape(4)
end

DIRS[argmax(prob)] #right is best move

model_group_cnn_cpu = model_group_cnn |> cpu

@time group_cnn_policy(one_game.states[pos])

@time [play_game_with_policy(group_cnn_policy, one_game.states[pos]) |> maximum for i in 1:1000] |> countmap
@time [play_game_with_policy(randompolicy, one_game.states[pos]) |> maximum for i in 1:1000] |> countmap

@time meh = Game2048.play_via_monte_carlo_w_recorder(randompolicy; n=1000, bitboard=one_game.states[pos])
meh.final_state
@time meh2 = play_via_monte_carlo_w_recorder(group_cnn_policy; n=1000, bitboard=one_game.states[pos])

# the rest is just checking to see how the recommendations change in relations to recommendations
best_move = prob |> onecold |> cpu |>
    (x->DIRS[x[1]])

# rotate left; up is now the best move should be right
one_game.states[pos] |>
    bitboard_to_array |>
    rotl90 |>
    transform_4corners_to_4cols |>
    (x->reshape(x, 4, 4, 1, :) ./ 11) |>
    cu |> model_group_cnn |> softmax

# up should be the best move
one_game.states[pos] |>
    bitboard_to_array |>
    rot180 |>
    transform_4corners_to_4cols |>
    (x->reshape(x, 4, 4, 1, :) ./ 11) |>
    cu |> model_group_cnn |> softmax

one_game.states[pos] |>
    bitboard_to_array |>
    rotr90 |>
    transform_4corners_to_4cols |>
    (x->reshape(x, 4, 4, 1, :) ./ 11) |>
    cu |> model_group_cnn |> softmax

#####################

function makey(state1)
    dir_cnt = zeros(Int, 4)
    for i in [Int(Game2048.find_best_move(state1, 1000, Game2048.randompolicy)) + 1 for i in 1:10]
        dir_cnt[i] += 1
    end
    dir_cnt./100
end

function normal_to_model(state1)
    @chain state1 begin
        bitboard_to_array
        transform_4corners_to_4cols
        reshape(_, 4, 4, 1, :) ./ 11
    end
end

state = gamerecord_nn.states[1]

model_group_cnn_cpu = model_group_cnn |> cpu

function model(x)
    # x = normal_to_model(state)

    xs = Ref(reshape(x,4,4)) .|> (identity, rotr90, rot180, rotl90) .|> (x->reshape(x, 4, 4, 1, :))

    result = model_group_cnn_cpu.(xs)

    for (i, r) in enumerate(result)
        r .= @view r[mod1.((1:4) .+ i, 4)]
    end

    # compute distance to mean
    rr = hcat(result...)
    map(eachrow(rr)) do row
        mean(row)
    end
end

loss(x, y) = Flux.logitbinarycrossentropy(model(x), y)

loss(x, [0.0, 1.0, 0.0, 0.0])

opt=ADAM()
@time Flux.train!(loss, Flux.params(model_group_cnn), [(x, [0.0, 1.0, 0.0, 0.0])], opt);


while true
    # play a game from scratch
    @time gamerecord_nn = play_via_monte_carlo_w_recorder(group_cnn_policy; n=1, bitboard=initbboard())

    move_probs = get_move_probs(gamerecord_nn)

    # look for the top 32 moves that need correction since computing every move is taking a long time
    # seems to be a better strategy to play 100 and then stop if maximum exceeds 1000
    # @time best_play = play_via_monte_carlo_w_recorder(randompolicy; n=1000, bitboard = gamerecord_nn.states[end-1])
    best_play100 = play_via_monte_carlo_w_recorder(randompolicy; n=100, bitboard = gamerecord_nn.states[end-1])
    while maximum(best_play100.final_state) < 11
        best_play100 = play_via_monte_carlo_w_recorder(randompolicy; n=100, bitboard = gamerecord_nn.states[end-1])
    end

    best_play100




    Flux.train!(loss, Flux.params(model_group_cnn), [(new_train_x, new_train_y)], opt);



    # use random play to figure out the best moves
    x = cat(normal_to_model.(gamerecord_nn.states)..., dims=4)
    wy = reduce(hcat, makey.(gamerecord_nn.states))

    obj = (gamerecorder = gamerecord_nn, x=x, y=y)
    hashofit = hash(obj)

    using Serialization: serialize
    # serialize(raw"C:\data\game2048\self-play-data\$hashofit", obj)

    @time Flux.train!(loss, Flux.params(model_group_cnn), [(cu(x), cu(y))], opt; cb); # compile

    model_group_cpu=model_group_cnn |> cpu
    pcpu = Flux.params(model_group_cpu)
    serialize("group_pcpu", pcpu)
    # pcpu = deserialize("group_pcpu")
    # Flux.loadparams!(model_group_cpu, pcpu)
    # model_group_cnn = model_group_cpu |> gpu
end
