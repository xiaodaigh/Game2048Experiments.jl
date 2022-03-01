export nn_policy_maker

using Flux: softmax

# an init_transform
function div_by_11(x)
    x ./ 11
end

"""
Given a neural work (nn) that accepts a 4x4x1xn array of inputs and returns
a policy
"""
function nn_policy_maker(nn, init_transform=div_by_11)
    function(board::Bitboard)
        tmp = Float32.(bitboard_to_array(board)) |> init_transform
        reshape(softmax(nn(reshape(tmp, 4, 4, 1, 1))), 4)
    end
end
