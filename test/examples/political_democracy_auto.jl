using StructuralEquationModels, Test, FiniteDiff, StenoGraphs
# import StructuralEquationModels as SEM

include("helpers.jl")

# test matrix
specification = [:partable, :ram_matrices]
build = [:constructor, :in_parts, :mixed]
model_type = [Sem, SemForwardDiff, SemFiniteDiff]
semdiff = [SemDiffOptim, SemDiffNLopt]
loss = [SemML, SemWLS]
imply = [RAM, RAMSymbolic]
#test = [:gradient, :solution, :fitmeasures, :se]

model_setups = Any[]

for s in specification
    for b in build
        for t in model_type
            for d in semdiff
                for l in loss
                    for i in imply
                        setup = Dict(
                            :specification => s,
                            :build => b,
                            :model_type => t,
                            :diff => d,
                            :loss => l,
                            :imply => i
                        )
                        push!(model_setups, setup)
                    end
                end
            end
        end
    end
end

build1 = Dict(
    :specification => partable,
    :build => :constructor,
    :model_type => Sem,
    :diff => SemDiffOptim,
    :loss => SemML,
    :imply => RAM
)

test_ml = Dict(
    :gradient => nothing,
    :solution => :parameter_estimates_ml,
    :fitmeasures => :fitmeasures_ml,
    :se => :parameter_estimates_ml
)

test_ls = Dict(
    :gradient => nothing,
    :solution => :parameter_estimates_ls,
    :fitmeasures => :fitmeasures_ls,
    :se => :parameter_estimates_ls
)

test_ml_mean = Dict(
    :gradient => nothing,
    :solution => :parameter_estimates_ml_mean,
    :fitmeasures => :fitmeasures_ml_mean,
    :se => :parameter_estimates_ml_mean
)

test_ls_mean = Dict(
    :gradient => nothing,
    :solution => :parameter_estimates_ls_mean,
    :fitmeasures => :fitmeasures_ls_mean,
    :se => :parameter_estimates_ls_mean
)

test_fiml = Dict(
    :gradient => nothing,
    :solution => :parameter_estimates_fiml,
    :fitmeasures => :fitmeasures_fiml,
    :se => :parameter_estimates_fiml
)

############################################################################
### data
############################################################################

dat = example_data("political_democracy")
dat_missing = example_data("political_democracy_missing")
solution = example_data("political_democracy_solution")

############################################################################
### specification
############################################################################

# graph -> partable
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

partable = ParameterTable(
    latent_vars = latent_vars,
    observed_vars = observed_vars,
    graph = graph)

# ram_matrices
x = Symbol.("x".*string.(1:31))

S =[:x1   0    0     0     0      0     0     0     0     0     0     0     0     0
    0     :x2  0     0     0      0     0     0     0     0     0     0     0     0
    0     0     :x3  0     0      0     0     0     0     0     0     0     0     0
    0     0     0     :x4  0      0     0     :x15  0     0     0     0     0     0
    0     0     0     0     :x5   0     :x16  0     :x17  0     0     0     0     0
    0     0     0     0     0     :x6  0      0     0     :x18  0     0     0     0
    0     0     0     0     :x16  0     :x7   0     0     0     :x19  0     0     0
    0     0     0     :x15 0      0     0     :x8   0     0     0     0     0     0
    0     0     0     0     :x17  0     0     0     :x9   0     :x20  0     0     0
    0     0     0     0     0     :x18 0      0     0     :x10  0     0     0     0
    0     0     0     0     0     0     :x19  0     :x20  0     :x11  0     0     0
    0     0     0     0     0     0     0     0     0     0     0     :x12  0     0
    0     0     0     0     0     0     0     0     0     0     0     0     :x13  0
    0     0     0     0     0     0     0     0     0     0     0     0     0     :x14]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  0  0     1.0   0     0
    0  0  0  0  0  0  0  0  0  0  0     :x21  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x22  0     0
    0  0  0  0  0  0  0  0  0  0  0     0     1.0   0
    0  0  0  0  0  0  0  0  0  0  0     0     :x23  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x24  0
    0  0  0  0  0  0  0  0  0  0  0     0     :x25  0
    0  0  0  0  0  0  0  0  0  0  0     0     0     1
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x26
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x27
    0  0  0  0  0  0  0  0  0  0  0     0     0     :x28
    0  0  0  0  0  0  0  0  0  0  0     0     0     0
    0  0  0  0  0  0  0  0  0  0  0     :x29  0     0
    0  0  0  0  0  0  0  0  0  0  0     :x30  :x31  0]

ram_matrices = RAMMatrices(;
    A = A, 
    S = S, 
    F = F, 
    parameters = x,
    colnames = [:x1, :x2, :x3, :y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8, :ind60, :dem60, :dem65])

function test_sem(;specification, build, model_type, diff, loss, imply, tests, kwargs...)
    # build
    if build == :constructor
        model = model_type(
            specification = specification,
            diff = diff,
            loss = loss,
            imply = imply
        )
    elseif build == :in_parts
        model = model_type()
    elseif build == :mixed

    end
    # test
    for test in keys(tests)
        @testset "" begin
            @test test[1](model, test[2])
        end
    end
end

specification = [:partable, :ram_matrices]
build = [:constructor, :in_parts, :mixed]
model_type = [Sem, SemForwardDiff, SemFiniteDiff]
diff = [SemDiffOptim, SemDiffNlopt]
loss = [SemML, SemWLS]
imply = [RAM, RAMSymbolic]
test = [:gradient, :solution, :fitmeasures, :se]



build1 = Dict(
    :specification => partable,
    :build => :constructor,
    :model_type => Sem,
    :diff => SemDiffOptim,
    :loss => SemML,
    :imply => RAM
)



setup1 = Dict(
    :build => build1,
    :test => test1
)

a = SemML

"$a"