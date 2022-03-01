# I want to try a pure MC simulation that doesn't rely on a neural network and
# see how things
using StatsBase: countmap
using Game2048Core
using BenchmarkTools

using Game2048Core: bitboard_to_array, DIRS

board = initbboard()

@benchmark simulate_bb(board)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):   4.500 μs …  6.343 ms  ┊ GC (min … max): 0.00% … 99.27%
#  Time  (median):     17.300 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   20.922 μs ± 99.104 μs  ┊ GC (mean ± σ):  8.10% ±  1.72%

#         ▂▅█▆▆█▇▇▄█▇▇▄▃▂▁▁
#   ▁▂▃▄▆▇███████████████████▇▆▆▅▅▅▄▄▄▃▄▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁ ▄
#   4.5 μs          Histogram: frequency by time        51.3 μs <

#  Memory estimate: 5.56 KiB, allocs estimate: 94.


@time play_via_monte_carlo(board, 50)

@time bitboard_to_array(play_via_monte_carlo(board, 1)) |> maximum


@time res = [bitboard_to_array(play_via_monte_carlo(board, 512)) |> maximum for i in 1:10]

@time countmap(res)


a = [maximum(simulate_bb(initbboard()) |> bitboard_to_array) for i in 1:1_000_000]
countmap(a)

