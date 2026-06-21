using SparseArrays, StructuralEquationModels, Test

const SEM = StructuralEquationModels

function toy_regularization_spec()
    A = [0 0 :lambda1; 0 0 :lambda2; 0 0 0]
    S = [:theta1 0 0; 0 :theta2 0; 0 0 :phi]
    F = [1.0 0.0 0.0; 0.0 1.0 0.0]

    return SEM.RAMMatrices(;
        A,
        S,
        F,
        param_labels = [:lambda1, :lambda2, :theta1, :theta2, :phi],
        vars = [:y1, :y2, :eta],
    )
end

@testset "regularization constructors share SemParamsPenalty" begin
    default_shape = SEM.SemParamsPenalty(1:2; shape2 = 2, weight2 = 0.75)
    default_elastic = SEM.SemElasticNet(1:2)
    spec = toy_regularization_spec()

    @test SEM.SemNorm(1:2) isa SEM.SemParamsPenalty
    @test SEM.SemNorm(1:2; shape = 3) isa SEM.SemParamsPenalty
    @test SEM.SemParamsPenalty(1:2; shape2 = 2, weight2 = 0.75) isa SEM.SemParamsPenalty
    @test SEM.SemParamsPenalty(1:2) isa SEM.SemParamsPenalty
    @test default_shape.shape1 == 1
    @test !(:weight1 in fieldnames(typeof(default_shape)))
    @test SEM.SemLasso(1:2) isa SEM.SemParamsPenalty
    @test SEM.SemRidge(1:2) isa SEM.SemParamsPenalty
    @test SEM.SemElasticNet(1:2) isa SEM.SemParamsPenalty
    @test default_elastic.shape1 == 1
    @test default_elastic.shape2 == 2
    @test default_elastic.weight2 == 0.5
    @test SEM.SemElasticNet(1:2; weight2 = 0.25).weight2 == 0.25
    @test SEM.SemHinge(1:2; bound = :u) isa SEM.SemParamsPenalty
    @test SEM.SemHinge(1:2; bound = :u, shape1 = 2) isa SEM.SemParamsPenalty
    @test SEM.SemHinge(spec, [:lambda1, :phi]; bound = :l) isa SEM.SemParamsPenalty
    @test_throws MethodError SEM.SemElasticNet(1:2; shape1 = 3)
    @test_throws MethodError SEM.SemElasticNet(1:2; shape2 = 4)
    @test_throws MethodError SEM.SemNorm(1:2; shape2 = 2)
    @test_throws MethodError SEM.SemNorm(1:2; weight = 0.25)
    @test_throws MethodError SEM.SemParamsPenalty(1:2; weight1 = 0.25)
    @test_throws TypeError SEM.SemParamsPenalty(1:2; shape1 = nothing)
end

@testset "SemParamsPenalty | generic fractional and lower hinge branches" begin
    fractional = SEM.SemParamsPenalty(1:2; shape1 = 1.5, shape2 = 3.0, weight2 = 0.5)
    params = [4.0, 1.0]
    grad = zeros(2)
    hess = zeros(2, 2)

    obj = SEM.evaluate!(0.0, grad, hess, fractional, params)

    @test fractional.shape1 == 1.5
    @test fractional.shape2 == 3
    @test fractional.weight2 == 0.5
    @test obj ≈ 41.5
    @test grad ≈ [27.0, 3.0]
    @test hess ≈ [12.375 0.0; 0.0 3.75]

    lower_hinge = SEM.SemHinge(1:3; bound = :l, threshold = 0.5, shape1 = 3)
    params = [0.0, 1.0, 2.0]
    grad = zeros(3)
    hess = zeros(3, 3)

    obj = SEM.evaluate!(0.0, grad, hess, lower_hinge, params)

    @test obj ≈ 3.5
    @test grad ≈ [0.0, 0.75, 6.75]
    @test hess ≈ [0.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 9.0]
end

@testset "SemParamsPenalty | transformed elastic net" begin
    penalty = SEM.SemParamsPenalty(
        nothing,
        [1.0 0.0 1.0; 0.0 1.0 0.0],
        [0.0, 1.0];
        shape2 = 2,
        weight2 = 0.5,
    )
    params = [1.0, -2.0, 3.0]
    grad = zeros(3)
    hess = zeros(3, 3)

    obj = SEM.evaluate!(0.0, grad, hess, penalty, params)

    @test obj ≈ 13.5
    @test grad ≈ [5.0, -2.0, 5.0]
    @test hess ≈ [1.0 0.0 1.0; 0.0 1.0 0.0; 1.0 0.0 1.0]
end

@testset "SemParamsPenalty | SemSpecification constructors" begin
    spec = toy_regularization_spec()

    selected = SEM.SemParamsPenalty(spec, [:lambda1, :phi]; shape1 = 2)
    params = [2.0, 7.0, 11.0, 13.0, -3.0]
    grad = zeros(5)
    hess = zeros(5, 5)

    obj = SEM.evaluate!(0.0, grad, hess, selected, params)

    expected_hess = zeros(5, 5)
    expected_hess[1, 1] = 2.0
    expected_hess[5, 5] = 2.0

    @test selected.param_inds === nothing
    @test size(selected.A) == (2, SEM.nparams(spec))
    @test SEM.nparams(selected) == SEM.nparams(spec)
    @test obj ≈ 13.0
    @test grad ≈ [4.0, 0.0, 0.0, 0.0, -6.0]
    @test hess ≈ expected_hess

    dense_transformed = SEM.SemParamsPenalty(spec, [:lambda1, :phi], [1.0 -1.0]; shape1 = 2)
    params = [1.0, 0.0, 0.0, 0.0, 3.0]
    grad = zeros(5)
    hess = zeros(5, 5)

    obj = SEM.evaluate!(0.0, grad, hess, dense_transformed, params)

    expected_hess = zeros(5, 5)
    expected_hess[1, 1] = 2.0
    expected_hess[1, 5] = -2.0
    expected_hess[5, 1] = -2.0
    expected_hess[5, 5] = 2.0

    @test dense_transformed.param_inds == [1, 5]
    @test size(dense_transformed.A) == (1, 2)
    @test SEM.nparams(dense_transformed) == 2
    @test obj ≈ 4.0
    @test grad ≈ [-4.0, 0.0, 0.0, 0.0, 4.0]
    @test hess ≈ expected_hess

    sparse_transformed =
        SEM.SemParamsPenalty(spec, [:lambda1, :phi], sparse([1.0 1.0]); shape1 = 1)
    params = [2.0, 0.0, 0.0, 0.0, -1.0]
    grad = zeros(5)
    hess = zeros(5, 5)

    obj = SEM.evaluate!(0.0, grad, hess, sparse_transformed, params)

    @test sparse_transformed.param_inds === nothing
    @test sparse_transformed.A isa SparseMatrixCSC
    @test size(sparse_transformed.A) == (1, SEM.nparams(spec))
    @test SEM.nparams(sparse_transformed) == SEM.nparams(spec)
    @test obj ≈ 1.0
    @test grad ≈ [1.0, 0.0, 0.0, 0.0, 1.0]
    @test hess == zeros(5, 5)

    @test_throws DimensionMismatch SEM.SemParamsPenalty(spec, [:lambda1, :phi], ones(1, 3))
end

@testset "SemParamsPenalty | lasso and hinge families" begin
    params = [-1.0, 0.5, 2.0]

    lasso = SEM.SemLasso(1:3)
    grad = zeros(3)
    hess = zeros(3, 3)
    obj = SEM.evaluate!(0.0, grad, hess, lasso, params)

    @test obj ≈ 3.5
    @test grad ≈ [-1.0, 1.0, 1.0]
    @test hess == zeros(3, 3)

    hinge = SEM.SemHinge(1:3; bound = :u, threshold = 0.0)
    fill!(grad, 0.0)
    fill!(hess, 0.0)
    obj = SEM.evaluate!(0.0, grad, hess, hinge, params)

    @test obj ≈ 1.0
    @test grad ≈ [-1.0, 0.0, 0.0]
    @test hess == zeros(3, 3)

    squared_hinge = SEM.SemHinge(1:3; bound = :u, threshold = 0.0, shape1 = 2)
    fill!(grad, 0.0)
    fill!(hess, 0.0)
    obj = SEM.evaluate!(0.0, grad, hess, squared_hinge, params)

    @test obj ≈ 1.0
    @test grad ≈ [-2.0, 0.0, 0.0]
    @test hess ≈ [2.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
end

@testset "weighted loss terms wrap positional penalties" begin
    ridge_term = SEM.LossTerm(SEM.SemRidge(1:2), nothing, 0.25)
    hinge_term = SEM.LossTerm(SEM.SemHinge(1:2; bound = :u, shape1 = 2), nothing, 0.5)

    @test ridge_term isa SEM.LossTerm
    @test hinge_term isa SEM.LossTerm
    @test SEM.weight(ridge_term) == 0.25
    @test SEM.weight(hinge_term) == 0.5
    @test SEM.loss(ridge_term) isa SEM.SemParamsPenalty
    @test SEM.loss(hinge_term) isa SEM.SemParamsPenalty
end
