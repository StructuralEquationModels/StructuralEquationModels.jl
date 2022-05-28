using StructuralEquationModels, Test, Statistics
import StructuralEquationModels: obs_cov, get_data

include(
    joinpath(chop(dirname(pathof(StructuralEquationModels)), tail = 3), 
    "test/examples/helper.jl")
    )

############################################################################################
### without meanstructure
############################################################################################

### model specification --------------------------------------------------------------------

observed_vars = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
latent_vars = [:ind60, :dem60, :dem65]

graph = @StenoGraph begin
    # loadings
    ind60 → fixed(1)*x1 + x2 + x3
    dem60 → fixed(1)*y1 + y2 + y3 + y4
    dem65 → fixed(1)*y5 + y6 + y7 + y8
    # latent regressions
    label(:a)*dem60 ← ind60
    dem65 ← dem60
    dem65 ← ind60
    # variances
    _(observed_vars) ↔ _(observed_vars)
    _(latent_vars) ↔ _(latent_vars)
    # covariances
    y1 ↔ y5
    y2 ↔ y4 + y6
    y3 ↔ y7
    y8 ↔ y4 + y6
end

spec = ParameterTable(
    latent_vars = latent_vars,
    observed_vars = observed_vars,
    graph = graph)

### data -----------------------------------------------------------------------------------

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")[:, names(dat)]

dat_matrix = Matrix(dat)
dat_missing_matrix = Matrix(dat_missing)

dat_cov = Statistics.cov(dat_matrix)

### tests - SemObsCommon -------------------------------------------------------------------

# test errors

@test_throws ArgumentError("please provide column names via the `data_colnames = ...` argument.") begin
    SemObsCommon(specification = spec, data = dat_matrix)
end

@test_throws ArgumentError("if an observed covariance is given, `data_colnames = ...` has to be specified.") begin
    SemObsCommon(specification = spec, obs_cov = dat_cov)
end

@test_throws ArgumentError("please specify `data_colnames` as a vector of Symbols") begin
    SemObsCommon(specification = spec, data = dat_matrix, data_colnames = names(dat))
end

@test_throws ArgumentError("you specified neither an observed dataset nor an observed covariance matrix") begin
    SemObsCommon(specification = spec)
end

@test_throws ArgumentError("you specified both an observed dataset and an observed covariance matrix") begin
    SemObsCommon(specification = spec, data = dat_matrix, obs_cov = dat_cov)
end

@test_throws UndefKeywordError SemObsCommon(data = dat_matrix)

@test_throws UndefKeywordError SemObsCommon(obs_cov = dat_cov)

# should work
observed = SemObsCommon(
    specification = spec,
    data = dat
)

observed_nospec = SemObsCommon(
    specification = nothing,
    data = dat_matrix
)

observed_matrix = SemObsCommon(
    specification = spec, 
    data = dat_matrix, 
    data_colnames = Symbol.(names(dat))
)

observed_cov = SemObsCommon(
    specification = spec,
    obs_cov = dat_cov,
    data_colnames = Symbol.(names(dat)),
    n_obs = 75.0
)

all_equal_cov = (obs_cov(observed) == obs_cov(observed_nospec)) &
            (obs_cov(observed) == obs_cov(observed_matrix)) &
            (obs_cov(observed) == obs_cov(observed_cov))

all_equal_data = (get_data(observed) == get_data(observed_nospec)) &
            (get_data(observed) == get_data(observed_matrix)) &
            (get_data(observed) !== get_data(observed_cov))

@testset "unit tests | SemObsCommon | input formats" begin
    @test all_equal_cov
    @test all_equal_data
    @test n_obs(observed) == n_obs(observed_cov)
end


# shuffle variables
new_order = [3,2,7,8,5,6,9,11,1,10,4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat = dat[:, new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]

shuffle_dat_cov = Statistics.cov(shuffle_dat_matrix)

observed_shuffle = SemObsCommon(
    specification = spec,
    data = shuffle_dat
)

observed_matrix_shuffle = SemObsCommon(
    specification = spec, 
    data = shuffle_dat_matrix, 
    data_colnames = shuffle_names
)

observed_cov_shuffle = SemObsCommon(
    specification = spec,
    obs_cov = shuffle_dat_cov,
    data_colnames = shuffle_names
)

all_equal_cov_suffled = (obs_cov(observed) == obs_cov(observed_shuffle)) &
            (obs_cov(observed) == obs_cov(observed_matrix_shuffle)) &
            (obs_cov(observed) ≈ obs_cov(observed_cov_shuffle))

all_equal_data_suffled = (get_data(observed) == get_data(observed_shuffle)) &
            (get_data(observed) == get_data(observed_matrix_shuffle)) &
            (get_data(observed) !== get_data(observed_cov_shuffle))

@testset "unit tests | SemObsCommon | input formats shuffled " begin
    @test all_equal_cov_suffled
    @test all_equal_data_suffled
end

### tests - SemObsMissing ------------------------------------------------------------------

# test errors

@test_throws ArgumentError("please provide column names via the `data_colnames = ...` argument.") begin
    SemObsMissing(specification = spec, data = dat_missing_matrix)
end

@test_throws ArgumentError("please specify `data_colnames` as a vector of Symbols") begin
    SemObsMissing(specification = spec, data = dat_missing_matrix, data_colnames = names(dat))
end

@test_throws UndefKeywordError(:data) begin
    SemObsMissing(specification = spec)
end

@test_throws UndefKeywordError SemObsMissing(data = dat_matrix)

# should work
observed = SemObsMissing(
    specification = spec,
    data = dat_missing
)

observed_nospec = SemObsMissing(
    specification = nothing,
    data = dat_missing
)

observed_matrix = SemObsMissing(
    specification = spec, 
    data = dat_missing_matrix, 
    data_colnames = Symbol.(names(dat))
)

all_equal_missing = 
    isequal(get_data(observed), get_data(observed_nospec)) &
    isequal(get_data(observed), get_data(observed_matrix))

@testset "unit tests | SemObsMissing | input formats" begin
    @test all_equal_missing
end

# shuffle variables

new_order = [3,2,7,8,5,6,9,11,1,10,4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat = dat_missing[:, new_order]

shuffle_dat_matrix = dat_missing_matrix[:, new_order]

observed_shuffle = SemObsMissing(
    specification = spec,
    data = shuffle_dat
)

observed_matrix_shuffle = SemObsMissing(
    specification = spec, 
    data = shuffle_dat_matrix, 
    data_colnames = shuffle_names
)

all_equal_suffled = 
    isequal(get_data(observed), get_data(observed_shuffle)) &
    isequal(get_data(observed), get_data(observed_matrix_shuffle))

@testset "unit tests | SemObsMissing | input formats shuffled " begin
    @test all_equal_suffled
end