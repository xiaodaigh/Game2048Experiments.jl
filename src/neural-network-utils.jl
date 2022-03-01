"""
    Apply a rotation CNN
"""
function transform_4corners_to_4cols(bb)
    top_left = bb[1:2, 1:2]
    top_right = [
            bb[1, 4] bb[2,4];
            bb[1, 3] bb[2,3]
        ]
    bottom_right =
        [
            bb[4, 4] bb[4,3];
            bb[3, 4] bb[3,3]
        ]
    bottom_left =
        [
            bb[4, 1] bb[3,1];
            bb[4, 2] bb[3,2]
        ]

    reduce(hcat, reshape.([top_left, top_right, bottom_left, bottom_right], :))
end