# data associated with the specific pattern of missing manifested variables
struct SemObservedMissingPattern{T,S}
    obs_mask::BitVector     # observed vars mask
    miss_mask::BitVector    # missing vars mask
    nobserved::Int
    nmissed::Int
    rows::Vector{Int}       # rows in original data
    data::Matrix{T}         # non-missing submatrix of data

    obs_mean::Vector{S} # means of observed vars
    obs_cov::Symmetric{S, Matrix{S}}  # covariance of observed vars
end

function SemObservedMissingPattern(
    obs_mask::BitVector,
    rows::AbstractVector{<:Integer},
    data::AbstractMatrix
)
    T = nonmissingtype(eltype(data))

    pat_data = convert(Matrix{T}, view(data, rows, obs_mask))
    if size(pat_data, 1) > 1
        pat_mean, pat_cov = mean_and_cov(pat_data, 1, corrected=false)
        @assert size(pat_cov) == (size(pat_data, 2), size(pat_data, 2))
    else
        pat_mean = reshape(pat_data[1, :], 1, :)
        pat_cov = fill(zero(T), 1, 1)
    end

    miss_mask = .!obs_mask

    return SemObservedMissingPattern{T, eltype(pat_mean)}(
        obs_mask, miss_mask,
        sum(obs_mask), sum(miss_mask),
        rows, pat_data,
        dropdims(pat_mean, dims=1), Symmetric(pat_cov))
end

n_man(pat::SemObservedMissingPattern) = length(pat.obs_mask)
n_obs(pat::SemObservedMissingPattern) = length(pat.rows)

nobserved_vars(pat::SemObservedMissingPattern) = pat.nobserved
nmissed_vars(pat::SemObservedMissingPattern) = pat.nmissed
