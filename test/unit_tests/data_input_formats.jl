using StructuralEquationModels, Test, Statistics
import StructuralEquationModels: obs_cov, obs_mean, get_data

### model specification --------------------------------------------------------------------

spec = ParameterTable(nothing)
spec.variables[:observed_vars] = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]
spec.variables[:latent_vars] = [:ind60, :dem60, :dem65]

### data -----------------------------------------------------------------------------------

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")[:, names(dat)]

dat_matrix = Matrix(dat)
dat_missing_matrix = Matrix(dat_missing)

dat_cov = Statistics.cov(dat_matrix)
dat_mean = vcat(Statistics.mean(dat_matrix, dims = 1)...)

############################################################################################
### tests - SemObsData
############################################################################################

# w.o. means -------------------------------------------------------------------------------

# errors
@test_throws ArgumentError(
    "You passed your data as a `DataFrame`, but also specified `obs_colnames`. "*
    "Please make shure the column names of your data frame indicate the correct variables "*
    "or pass your data in a different format.") begin
    SemObsData(specification = spec, data = dat, obs_colnames = Symbol.(names(dat)))
end

@test_throws ArgumentError(
    "Your `data` can not be indexed by symbols. "*
    "Maybe you forgot to provide column names via the `obs_colnames = ...` argument.") begin
    SemObsData(specification = spec, data = dat_matrix)
end

@test_throws ArgumentError("please specify `obs_colnames` as a vector of Symbols") begin
    SemObsData(specification = spec, data = dat_matrix, obs_colnames = names(dat))
end

@test_throws UndefKeywordError(:data) SemObsData(specification = spec)

@test_throws UndefKeywordError(:specification) SemObsData(data = dat_matrix)

# should work
observed = SemObsData(
    specification = spec,
    data = dat
)

observed_nospec = SemObsData(
    specification = nothing,
    data = dat_matrix
)

observed_matrix = SemObsData(
    specification = spec, 
    data = dat_matrix, 
    obs_colnames = Symbol.(names(dat))
)

all_equal_cov = 
    (obs_cov(observed) == obs_cov(observed_nospec)) &
    (obs_cov(observed) == obs_cov(observed_matrix))

all_equal_data = 
    (get_data(observed) == get_data(observed_nospec)) &
    (get_data(observed) == get_data(observed_matrix))

@testset "unit tests | SemObsData | input formats" begin
    @test all_equal_cov
    @test all_equal_data
end

# shuffle variables
new_order = [3,2,7,8,5,6,9,11,1,10,4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat = dat[:, new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]

observed_shuffle = SemObsData(
    specification = spec,
    data = shuffle_dat
)

observed_matrix_shuffle = SemObsData(
    specification = spec, 
    data = shuffle_dat_matrix, 
    obs_colnames = shuffle_names
)

all_equal_cov_suffled = 
    (obs_cov(observed) == obs_cov(observed_shuffle)) &
    (obs_cov(observed) == obs_cov(observed_matrix_shuffle))

all_equal_data_suffled = 
    (get_data(observed) == get_data(observed_shuffle)) &
    (get_data(observed) == get_data(observed_matrix_shuffle))

@testset "unit tests | SemObsData | input formats shuffled " begin
    @test all_equal_cov_suffled
    @test all_equal_data_suffled
end

# with means -------------------------------------------------------------------------------

# errors
@test_throws ArgumentError(
    "You passed your data as a `DataFrame`, but also specified `obs_colnames`. "*
    "Please make shure the column names of your data frame indicate the correct variables "*
    "or pass your data in a different format.") begin
    SemObsData(specification = spec, data = dat, obs_colnames = Symbol.(names(dat)), meanstructure = true)
end

@test_throws ArgumentError(
    "Your `data` can not be indexed by symbols. "*
    "Maybe you forgot to provide column names via the `obs_colnames = ...` argument.") begin
    SemObsData(specification = spec, data = dat_matrix, meanstructure = true)
end

@test_throws ArgumentError("please specify `obs_colnames` as a vector of Symbols") begin
    SemObsData(specification = spec, data = dat_matrix, obs_colnames = names(dat), meanstructure = true)
end

@test_throws UndefKeywordError(:data) SemObsData(specification = spec, meanstructure = true)

@test_throws UndefKeywordError(:specification) SemObsData(data = dat_matrix, meanstructure = true)

# should work
observed = SemObsData(
    specification = spec,
    data = dat,
    meanstructure = true
)

observed_nospec = SemObsData(
    specification = nothing,
    data = dat_matrix,
    meanstructure = true
)

observed_matrix = SemObsData(
    specification = spec, 
    data = dat_matrix, 
    obs_colnames = Symbol.(names(dat)),
    meanstructure = true
)

all_equal_mean = 
    (obs_mean(observed) == obs_mean(observed_nospec)) &
    (obs_mean(observed) == obs_mean(observed_matrix))

@testset "unit tests | SemObsData | input formats - means" begin
    @test all_equal_mean
end

# shuffle variables
new_order = [3,2,7,8,5,6,9,11,1,10,4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat = dat[:, new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]

observed_shuffle = SemObsData(
    specification = spec,
    data = shuffle_dat,
    meanstructure = true
)

observed_matrix_shuffle = SemObsData(
    specification = spec, 
    data = shuffle_dat_matrix, 
    obs_colnames = shuffle_names,
    meanstructure = true
)

all_equal_mean_suffled = 
    (obs_mean(observed) == obs_mean(observed_shuffle)) &
    (obs_mean(observed) == obs_mean(observed_matrix_shuffle)) 


@testset "unit tests | SemObsData | input formats shuffled - mean" begin
    @test all_equal_mean_suffled
end

############################################################################################
### tests - SemObsCovariance
############################################################################################

# w.o. means -------------------------------------------------------------------------------

# errors

@test_throws ArgumentError("observed means were passed, but `meanstructure = false`") begin
    SemObsCovariance(specification = nothing, obs_cov = dat_cov, obs_mean = dat_mean)
end

@test_throws UndefKeywordError(:specification) SemObsCovariance(obs_cov = dat_cov)

@test_throws ArgumentError("no `obs_colnames` were specified") begin
    SemObsCovariance(specification = spec, obs_cov = dat_cov)
end

@test_throws ArgumentError("please specify `obs_colnames` as a vector of Symbols") begin
    SemObsCovariance(specification = spec, obs_cov = dat_cov, obs_colnames = names(dat))
end

# should work
observed = SemObsCovariance(
    specification = spec,
    obs_cov = dat_cov,
    obs_colnames = obs_colnames = Symbol.(names(dat))
)

observed_nospec = SemObsCovariance(
    specification = nothing,
    obs_cov = dat_cov,
    n_obs = 75.0
)


all_equal_cov = (obs_cov(observed) == obs_cov(observed_nospec))

@testset "unit tests | SemObsCovariance | input formats" begin
    @test all_equal_cov
    @test isnothing(n_obs(observed))
    @test n_obs(observed_nospec) == 75.0
end


# shuffle variables
new_order = [3,2,7,8,5,6,9,11,1,10,4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]

shuffle_dat_cov = Statistics.cov(shuffle_dat_matrix)

observed_shuffle = SemObsCovariance(
    specification = spec,
    obs_cov = shuffle_dat_cov,
    obs_colnames = shuffle_names
)

all_equal_cov_suffled = (obs_cov(observed) ≈ obs_cov(observed_shuffle))

@testset "unit tests | SemObsCovariance | input formats shuffled " begin
    @test all_equal_cov_suffled
end

# with means -------------------------------------------------------------------------------

# errors
@test_throws ArgumentError("`meanstructure = true`, but no observed means were passed") begin
    SemObsCovariance(specification = spec, obs_cov = dat_cov, meanstructure = true)
end

@test_throws UndefKeywordError SemObsCovariance(data = dat_matrix, meanstructure = true)

@test_throws UndefKeywordError SemObsCovariance(obs_cov = dat_cov, meanstructure = true)

@test_throws ArgumentError("`meanstructure = true`, but no observed means were passed") begin
    SemObsCovariance(
        specification = spec,
        obs_cov = dat_cov,
        obs_colnames = Symbol.(names(dat)),
        meanstructure = true
    )
end

# should work
observed = SemObsCovariance(
    specification = spec,
    obs_cov = dat_cov,
    obs_mean = dat_mean,
    obs_colnames = Symbol.(names(dat)),
    n_obs = 75.0,
    meanstructure = true
)

observed_nospec = SemObsCovariance(
    specification = nothing,
    obs_cov = dat_cov,
    obs_mean = dat_mean,
    meanstructure = true,
    n_obs = 75.0
)

all_equal_mean = (obs_mean(observed) == obs_mean(observed_nospec))

@testset "unit tests | SemObsCovariance | input formats - means" begin
    @test all_equal_mean
end

# shuffle variables
new_order = [3,2,7,8,5,6,9,11,1,10,4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat = dat[:, new_order]

shuffle_dat_matrix = dat_matrix[:, new_order]

shuffle_dat_cov = Statistics.cov(shuffle_dat_matrix)
shuffle_dat_mean = vcat(Statistics.mean(shuffle_dat_matrix, dims = 1)...)

observed_shuffle = SemObsCovariance(
    specification = spec,
    obs_cov = shuffle_dat_cov,
    obs_mean = shuffle_dat_mean,
    obs_colnames = shuffle_names,
    n_obs = 75.0,
    meanstructure = true
)


all_equal_mean_suffled = (obs_mean(observed) == obs_mean(observed_shuffle))

@testset "unit tests | SemObsCovariance | input formats shuffled - mean" begin
    @test all_equal_mean_suffled
end

############################################################################################
### tests - SemObsMissing
############################################################################################

# errors
@test_throws ArgumentError(
    "You passed your data as a `DataFrame`, but also specified `obs_colnames`. "*
    "Please make shure the column names of your data frame indicate the correct variables "*
    "or pass your data in a different format.") begin
    SemObsMissing(specification = spec, data = dat_missing, obs_colnames = Symbol.(names(dat)))
end

@test_throws ArgumentError(
    "Your `data` can not be indexed by symbols. "*
    "Maybe you forgot to provide column names via the `obs_colnames = ...` argument.") begin
    SemObsMissing(specification = spec, data = dat_missing_matrix)
end

@test_throws ArgumentError("please specify `obs_colnames` as a vector of Symbols") begin
    SemObsMissing(specification = spec, data = dat_missing_matrix, obs_colnames = names(dat))
end

@test_throws UndefKeywordError(:data) SemObsMissing(specification = spec)

@test_throws UndefKeywordError(:specification) SemObsMissing(data = dat_missing_matrix)

# should work
observed = SemObsMissing(
    specification = spec,
    data = dat_missing
)

observed_nospec = SemObsMissing(
    specification = nothing,
    data = dat_missing_matrix
)

observed_matrix = SemObsMissing(
    specification = spec, 
    data = dat_missing_matrix, 
    obs_colnames = Symbol.(names(dat))
)

all_equal_data = 
    isequal(get_data(observed), get_data(observed_nospec)) &
    isequal(get_data(observed), get_data(observed_matrix))

@testset "unit tests | SemObsMissing | input formats" begin
    @test all_equal_data
end

# shuffle variables
new_order = [3,2,7,8,5,6,9,11,1,10,4]

shuffle_names = Symbol.(names(dat))[new_order]

shuffle_dat_missing = dat_missing[:, new_order]

shuffle_dat_missing_matrix = dat_missing_matrix[:, new_order]

observed_shuffle = SemObsMissing(
    specification = spec,
    data = shuffle_dat_missing
)

observed_matrix_shuffle = SemObsMissing(
    specification = spec, 
    data = shuffle_dat_missing_matrix, 
    obs_colnames = shuffle_names
)

all_equal_data_shuffled = 
    isequal(get_data(observed), get_data(observed_shuffle)) &
    isequal(get_data(observed), get_data(observed_matrix_shuffle))

@testset "unit tests | SemObsMissing | input formats shuffled " begin
    @test all_equal_data_suffled
end