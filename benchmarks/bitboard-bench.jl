# conclusion the array_board moves are very slow compared to bitboard

using Game2048
using Game2048: Bitboard, initboard
using BenchmarkTools


@benchmark move(board, LEFT) setup=(board=Bitboard(rand(UInt64)))
# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     2.000 ns (0.00% GC)
#   median time:      2.100 ns (0.00% GC)
#   mean time:        2.514 ns (0.00% GC)
#   maximum time:     69.200 ns (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1000


@benchmark move(board, left) setup=(board=Bitboard(rand(UInt64)))
# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     3.600 ns (0.00% GC)
#   median time:      3.700 ns (0.00% GC)
#   mean time:        4.308 ns (0.00% GC)
#   maximum time:     30.900 ns (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1000

@benchmark move!(board, left) setup=(board=initboard())
# BenchmarkTools.Trial:
#   memory estimate:  48 bytes
#   allocs estimate:  2
#   --------------
#   minimum time:     245.036 ns (0.00% GC)
#   median time:      279.661 ns (0.00% GC)
#   mean time:        292.479 ns (0.71% GC)
#   maximum time:     5.629 μs (93.97% GC)
#   --------------
#   samples:          10000
#   evals/sample:     413


@benchmark move(board, right) setup=(board=Bitboard(rand(UInt64)))
# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     3.500 ns (0.00% GC)
#   median time:      3.500 ns (0.00% GC)
#   mean time:        4.061 ns (0.00% GC)
#   maximum time:     35.900 ns (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1000

@benchmark move(board, up) setup=(board=Bitboard(rand(UInt64)))
# BenchmarkTools.Trial:
#   memory estimate:  192 bytes
#   allocs estimate:  2
#   --------------
#   minimum time:     75.154 ns (0.00% GC)
#   median time:      85.421 ns (0.00% GC)
#   mean time:        98.508 ns (8.79% GC)
#   maximum time:     2.728 μs (95.36% GC)
#   --------------
#   samples:          10000
#   evals/sample:     974

@benchmark move(board, down) setup=(board=Bitboard(rand(UInt64)))
# BenchmarkTools.Trial:
#   memory estimate:  192 bytes
#   allocs estimate:  2
#   --------------
#   minimum time:     75.334 ns (0.00% GC)
#   median time:      88.181 ns (0.00% GC)
#   mean time:        101.071 ns (8.87% GC)
#   maximum time:     2.872 μs (96.13% GC)
#   --------------
#   samples:          10000
#   evals/sample:     973
