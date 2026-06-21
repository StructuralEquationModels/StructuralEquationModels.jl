using StructuralEquationModels, Test

const SEM = StructuralEquationModels

@testset "regularization constructors share SemParamsPenalty" begin
    default_shape = SEM.SemParamsPenalty(1:2; shape2 = 2, weight2 = 0.75)

    @test SEM.SemNorm(1:2) isa SEM.SemParamsPenalty
    @test SEM.SemNorm(1:2; shape = 3) isa SEM.SemParamsPenalty
    @test SEM.SemParamsPenalty(1:2; shape2 = 2, weight2 = 0.75) isa
          SEM.SemParamsPenalty
    @test SEM.SemParamsPenalty(1:2) isa SEM.SemParamsPenalty
    @test default_shape.shape1 == 1
    @test !(:weight1 in fieldnames(typeof(default_shape)))
    @test SEM.SemLasso(1:2) isa SEM.SemParamsPenalty
    @test SEM.SemRidge(1:2) isa SEM.SemParamsPenalty
    @test SEM.SemHinge(1:2; bound = :u) isa SEM.SemParamsPenalty
    @test SEM.SemHinge(1:2; bound = :u, shape1 = 2) isa SEM.SemParamsPenalty
    @test_throws MethodError SEM.SemNorm(1:2; shape2 = 2)
    @test_throws MethodError SEM.SemNorm(1:2; weight = 0.25)
    @test_throws MethodError SEM.SemParamsPenalty(1:2; weight1 = 0.25)
    @test_throws TypeError SEM.SemParamsPenalty(1:2; shape1 = nothing)
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