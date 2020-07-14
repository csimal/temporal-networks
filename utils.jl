

"""
    bissection_search(xs, x)

Find the index `k` such that `xs[k]` <= `x` < `xs[k+1]`, assuming `xs` is sorted in increasing order.

Returns `0` if `x` is smaller than `xs[1]`. Runs in O(log N), where N is the length of `xs`.
"""
function bissection_search(xs::Vector{T}, x::T) where T <: Real
    if x < xs[1]
        return 0
    end
    if x >= xs[length(xs)]
        return length(xs)
    end
    min = 1
    max = length(xs)
    while min != max-1
        mid = div(min+max, 2)
        if xs[mid] <= x
            min = mid
        else
            max = mid
        end
    end
    return min
end
