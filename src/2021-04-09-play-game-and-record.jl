using Game2048
using Serialization: serialize
using Game2048: value, play_via_monte_carlo_w_recorder, play_via_monte_carlo_wo_recorder


# @time x = [play_via_monte_carlo_wo_recorder(randompolicy; n=1000) for _ in 1:10]

maximum.(x)

while true
    results = play_n_game(randompolicy;  ntries_per_turn=1000)
    objhash=hash(results)
    serialize("c:/data/game2048/raw/game2048_$objhash.jls", results)
end

# find the score

scores = map(x->value(x.final_state), results)
fnlmax = map(x->maximum(x.final_state), results)

using DataFrames, Chain, DataFrameMacros, Statistics

@chain DataFrame(;scores, fnlmax) begin
    groupby(:fnlmax)
    @combine(mean(:scores))
end


y = map(results) do x
    maximum(x.final_state)
end

countmap(y)


@time results_cnn = play_n_game(policy, 100)

@time play_n_game(policy, 1)

y_cnn = map(results_cnn) do x
    maximum(x.final_state)
end

countmap(y_cnn)

using Alert
alert()
