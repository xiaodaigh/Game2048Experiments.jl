# This is a pure implementation of traversals
using Game2048Core
using Game2048Core: DIRS, Bitboard, count0, simulate_bb, bitboard_to_array
# using Game2048: print_score, print_board
using BenchmarkTools
using StatsBase

const MASK = UInt8(15)

# should be
mutable struct TreeNode
    parent::Union{TreeNode,Nothing}
    board::Bitboard
    children::Vector{TreeNode}
    children_are_moves::Bool
    TreeNode(parent, board, children_are_moves) = new(parent, board, [], children_are_moves)
end

Base.print(tn::TreeNode) = print(tn.board)
Base.show(tn::TreeNode) = show(tn.board)
Base.display(tn::TreeNode) = display(tn.board)

function largest_val_in_corner(bitboard::Bitboard)
    max_tile = maximum(bitboard)
    max_tile_in_corner = (bitboard.board & MASK) == max_tile
    max_tile_in_corner
end

function largest_val_in_corner(tn::TreeNode)
    largest_val_in_corner(tn.board)
end

function add_children!(tn::TreeNode)
    if tn.children_are_moves
        candidates = move.(Ref(tn.board), DIRS)
        diff_to_before = Ref(tn.board) .!= candidates
        # at_least_1_cand = map(largest_val_in_corner, candidates) .& diff_to_before

        # if any(at_least_1_cand)
        #     tn.children = [TreeNode(tn, b, false) for (b, addit) in zip(candidates, at_least_1_cand) if addit]
        # else
        #     tn.children = [TreeNode(tn, b, false) for (b, addit) in zip(candidates, diff_to_before) if addit]
        # end
        tn.children = [TreeNode(tn, b, false) for (b, ok) in zip(candidates, diff_to_before) if ok]
    else
        # if you are here this means that your parent was a move
        # so do not move if you are the same as your parent
        if tn.board == tn.parent.board
            return
        end
        # count the number of 0 spots
        n = count0(tn.board)
        tn.children = reshape([TreeNode(tn, add_tile(tn.board, i, tf), true) for (i, tf) in Iterators.product(1:n, 1:2)], :)
        # tn.children = [TreeNode(tn, add_tile(tn.board, i, 1), true) for i in 1:n]
    end
end

function grow!(tn::TreeNode, levels)
    if levels == 0
        return
    end
    add_children!(tn)
    for c in tn.children
        grow!(c, levels - 1)
    end
end

function grow(board::Bitboard, args...)
    tn = TreeNode(nothing, board, true)
    grow!(tn, args...)
    tn
end

# find number of nodes
function count_nodes(tn::TreeNode)
    if length(tn.children) == 0
        return 1
    else
        return mapreduce(count_nodes, +, tn.children)
    end
end

function max2(f, itrs)
    reduce(@view itrs[3:end]; init = (f(itrs[1]), f(itrs[2]))) do (l1, l2), new_val
        fnv = f(new_val)
        fnv > l1 ? (fnv, l1) : (l1, max(fnv, l2))
    end
end

function sum2(bitboard::Bitboard)
    mapreduce(+, 0:4:60) do s
        1 << ((bitboard.board >> s) & MASK)
    end
end

function make_value(nsims)
    function value(bitboard::Bitboard)
        # # is the maximum tile in the corner
        # nsims = 100
        mapreduce(+, 1:nsims) do _
            # 1 << maximum(simulate_bb(bitboard))
            sb = simulate_bb(bitboard)
            sum2(sb) + 10 * maximum(sb)
        end / nsims

        # # max_tile_in_corner = mapreduce(|, (0, 12, 48, 60)) do s
        # #     val = (bitboard.board >> s) & MASK
        # #     val == max_tile
        # # end

        # max_tile_in_corner = (bitboard.board & MASK) == max_tile
        # next_tile_is_also_good = ((bitboard.board >> 4) & MASK) in (max_tile, max_tile - 1)

        # val_add = next_tile_is_also_good ? 5 : 4

        # # map((0, 12, 48, 60)) do s
        # #     (bitboard.board >> s) & MASK
        # # end .|> Int

        # val2 = max_tile_in_corner ? val_add : 1

        # # val1 = mapreduce(+, 0:4:60) do s
        # #     power = (bitboard.board >> s) & MASK
        # #     1 << power
        # # end

        # c0 = count0(bitboard)

        # return (1 << max_tile) * (1 + c0 / 16) * val2
        # return
    end
end


# simulate each node 10 times
function get_score(tn::TreeNode, value)::Float64
    if length(tn.children) == 0
        return Float64(value(tn.board))
    else
        if tn.children_are_moves
            return maximum(c -> get_score(c, value), tn.children)
        else
            scores = get_score.(tn.children, Ref(value))
            l = length(scores)
            @assert iseven(l)
            return mean(@view scores[1:div(l, 2)]) * 0.9 + mean(@view scores[div(l, 2)+1:l]) * 0.1
        end
    end
end

function grow_nodes!(tn::TreeNode, n)
    if length(tn.children) == 0
        if tn.children_are_moves
            grow!(tn, n)
            return
        else
            return
        end
    else
        grow_nodes!.(tn.children, n)
    end
    return
end

function play(value)
    b = initbboard()
    tn = grow(b, 2)

    # count_nodes(tn)

    while true
        best_child = argmax(c -> get_score(c, value), tn.children)

        if best_child.board == tn.board
            return tn
        end

        # add a tile
        new_board = add_tile(best_child.board)

        for c in best_child.children
            if c.board == new_board
                tn = c
                break
            end
        end

        # cntn = count_nodes(tn)
        # if cntn < 10_000
        grow_nodes!(tn, 2)
        # end
        # cntn = count_nodes(tn)

        # if cntn < 10_000
        #     grow_nodes!(tn, 2)
        # end

        # println(count_nodes(tn))
        # println(cntn)
        # display(tn)

        if length(tn.children) == 0
            return tn
        end
    end
end

value20 = make_value(20)
@time tn=play(value20)

@time res_prefer_corner = [play(value20).board for _ in 1:10];

a = DataFrame(maxtile = maximum.(res_prefer_corner), score = Game2048Core.score.(res_prefer_corner))
# prefering corner in a simplistic way makes the score worse

if false
    svalues = [make_value(i) for i in 10:10:100]

    @time play(svalues[2])
    res = Dict{Tuple{Int,Int},Bitboard}()

    using ProgressMeter
    parts = sum([10 * i for i in 1:10])
    p = Progress(parts)

    @time Threads.@threads for i in 11:20
        Threads.@threads for j in 1:10
            a = play(svalues[j])
            res[(i, j)] = a.board
            for _ in 1:j
                next!(p)
            end
        end
    end

    # @time res1 = [play().board for _ in 1:10]

    @time res2 = [maximum(r) |> Int for r in res]

    @time res2 = [Game2048Core.score(r) |> Int for r in res]

    using DataFrames, DataFrameMacros, Chain
    a = DataFrame(k = keys(res) |> collect, v = [v for (_, v) in res])

    a1 = @chain a begin
        @transform begin
            :try_id = :k[1]
            :nsims = :k[2] * 10
            :score = Game2048Core.score(:v)
            :maxtile = maximum(:v)
        end
    end

    @chain a1 begin
        @transform :win = :maxtile >= 11
        groupby(:nsims)
        @combine(std(:score), mean(:score), mean(:win))
    end

    @chain a1 begin
        @subset :nsims == 90
        sort!(:score, rev = true)
    end

    @chain a1 begin
        @subset :nsims == 100
        sort!(:score, rev = true)
    end

    using Serialization

    serialize("./data/runs_from_sims_10_to_100.csv", a1)

    mean(res2)
    # 2886.4 for 10 trieson
    # 4410.8 for 20 trieson

    countmap(res2)

    # value(bb)

    # @time res = [simulate_bb() for _ in 1:1000]
    # x = res[1]

    # @time tn = grow(b, 6);
    # @time count_nodes(tn)
    # tn.board
    # value(tn.board)
    # @time get_score.(tn.children)

    # @time res = play()
end