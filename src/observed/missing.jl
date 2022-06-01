############################################################################################
### Types
############################################################################################

# Type to store Expectation Maximization result --------------------------------------------
mutable struct EmMVNModel{A, b, B}
    Σ::A
    μ::b
    fitted::B
end

"""
For observed data with missing values.

# Constructor

    SemObsMissing(;
        specification,
        data,
        obs_colnames = nothing,
        kwargs...)

# Arguments
- `specification`: either a `RAMMatrices` or `ParameterTable` object (1)
- `data`: observed data
- `obs_colnames::Vector{Symbol}`: column names of the data (if the object passed as data does not have column names, i.e. is not a data frame)

# Extended help
## Interfaces
- `n_obs(::SemObsMissing)` -> number of observed data points
- `n_man(::SemObsMissing)` -> number of manifest variables

- `get_data(::SemObsMissing)` -> observed data
- `data_rowwise(::SemObsMissing)` -> observed data as vector per observation, with missing values deleted

- `patterns(::SemObsMissing)` -> indices of non-missing variables per missing patterns 
- `patterns_not(::SemObsMissing)` -> indices of missing variables per missing pattern
- `rows(::SemObsMissing)` -> row indices of observed data points that belong to each pattern
- `pattern_n_obs(::SemObsMissing)` -> number of data points per pattern
- `pattern_nvar_obs(::SemObsMissing)` -> number of non-missing observed variables per pattern
- `obs_mean(::SemObsMissing)` -> observed mean per pattern
- `obs_cov(::SemObsMissing)` -> observed covariance per pattern
- `em_model(::SemObsMissing)` -> `EmMVNModel` that contains the covariance matrix and mean vector found via optimization maximization

## Implementation
Subtype of `SemObs`

## Remarks
(1) the `specification` argument can also be `nothing`, but this turns of checking whether
the observed data/covariance columns are in the correct order! As a result, you should only
use this if you are shure your observed data is in the right format.

## Additional keyword arguments:
- `spec_colnames::Vector{Symbol} = nothing`: overwrites column names of the specification object
"""
mutable struct SemObsMissing{
        A <: AbstractArray,
        D <: AbstractFloat,
        O <: AbstractFloat,
        P <: Vector,
        P2 <: Vector,
        R <: Vector,
        PD <: AbstractArray,
        PO <: AbstractArray,
        PVO <: AbstractArray,
        A2 <: AbstractArray,
        A3 <: AbstractArray,
        S <: EmMVNModel
        } <: SemObs
    data::A
    n_man::D
    n_obs::O
    patterns::P # missing patterns
    patterns_not::P2
    rows::R # coresponding rows in data_rowwise
    data_rowwise::PD # list of data
    pattern_n_obs::PO # observed rows per pattern
    pattern_nvar_obs::PVO # number of non-missing variables per pattern
    obs_mean::A2
    obs_cov::A3
    em_model::S
end

############################################################################################
### Constructors
############################################################################################

function SemObsMissing(;
        specification,
        data,

        obs_colnames = nothing,
        spec_colnames = nothing,

        kwargs...)

    if isnothing(spec_colnames) spec_colnames = get_colnames(specification) end

    if !isnothing(spec_colnames)
        if isnothing(obs_colnames)
            try
                data = data[:, spec_colnames]
            catch
                throw(ArgumentError(
                    "Your `data` can not be indexed by symbols. "*
                    "Maybe you forgot to provide column names via the `obs_colnames = ...` argument.")
                    )
            end
        else
            if data isa DataFrame
                throw(ArgumentError(
                    "You passed your data as a `DataFrame`, but also specified `obs_colnames`. "*
                    "Please make shure the column names of your data frame indicate the correct variables "*
                    "or pass your data in a different format.")
                    )
            end

            if !(eltype(obs_colnames) <: Symbol)
                throw(ArgumentError("please specify `obs_colnames` as a vector of Symbols"))
            end

            data = reorder_data(data, spec_colnames, obs_colnames)
        end
    end

    if data isa DataFrame
        data = Matrix(data)
    end

    # remove persons with only missings
    keep = Vector{Int64}()
    for i = 1:size(data, 1)
        if any(.!ismissing.(data[i, :]))
            push!(keep, i)
        end
    end
    data = data[keep, :]



    n_obs, n_man = size(data)

    # compute and store the different missing patterns with their rowindices
    missings = ismissing.(data)
    patterns = [missings[i, :] for i = 1:size(missings, 1)]

    patterns_cart = findall.(!, patterns)
    data_rowwise = [data[i, patterns_cart[i]] for i = 1:n_obs]
    data_rowwise = convert.(Array{Float64}, data_rowwise)

    remember = Vector{BitArray{1}}()
    rows = [Vector{Int64}(undef, 0) for i = 1:size(patterns, 1)]
    for i = 1:size(patterns, 1)
        unknown = true
        for j = 1:size(remember, 1)
            if patterns[i] == remember[j]
                push!(rows[j], i)
                unknown = false
            end
        end
        if unknown
            push!(remember, patterns[i])
            push!(rows[size(remember, 1)], i)
        end
    end
    rows = rows[1:length(remember)]
    n_patterns = size(rows, 1)

    # sort by number of missings
    sort_n_miss = sortperm(sum.(remember))
    remember = remember[sort_n_miss]
    remember_cart = findall.(!, remember)
    remember_cart_not = findall.(remember)
    rows = rows[sort_n_miss]

    pattern_n_obs = size.(rows, 1)
    pattern_nvar_obs = length.(remember_cart) 

    cov_mean = [cov_and_mean(data_rowwise[rows]) for rows in rows]
    obs_cov = [cov_mean[1] for cov_mean in cov_mean]
    obs_mean = [cov_mean[2] for cov_mean in cov_mean]

    em_model = EmMVNModel(zeros(n_man, n_man), zeros(n_man), false)

    return SemObsMissing(data, Float64(n_man), Float64(n_obs), remember_cart,
    remember_cart_not, 
    rows, data_rowwise, Float64.(pattern_n_obs), Float64.(pattern_nvar_obs),
    obs_mean, obs_cov, em_model)
end

############################################################################################
### Recommended methods
############################################################################################

n_obs(observed::SemObsMissing) = observed.n_obs
n_man(observed::SemObsMissing) = observed.n_man

############################################################################################
### Additional methods
############################################################################################

get_data(observed::SemObsMissing) = observed.data
patterns(observed::SemObsMissing) = observed.patterns
patterns_not(observed::SemObsMissing) = observed.patterns_not
rows(observed::SemObsMissing) = observed.rows
data_rowwise(observed::SemObsMissing) = observed.data_rowwise
pattern_n_obs(observed::SemObsMissing) = observed.pattern_n_obs
pattern_nvar_obs(observed::SemObsMissing) = observed.pattern_nvar_obs
obs_mean(observed::SemObsMissing) = observed.obs_mean
obs_cov(observed::SemObsMissing) = observed.obs_cov
em_model(observed::SemObsMissing) = observed.em_model