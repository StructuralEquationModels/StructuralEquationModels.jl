mutable struct model{
        RAM <: Function,
        DATA <: Union{Matrix{Float64}, Nothing},
        PAR <: Union{Array{Float64, 1}, Nothing},
        MSTRUC <: Union{Bool, Nothing},
        LOGL <: Union{Float64, Nothing},
        OPT <: Union{String, Nothing},
        EST <: Union{Function, Nothing},
        OBS_COV <: Union{Matrix{Float64}, Nothing},
        IMP_COV <: Union{Matrix{Float64}, Nothing},
        OBS_MEAN <: Union{Array{Float64, 2}, Nothing},
        OPT_RESULT <: Any,
        SE <: Union{Array{Float64, 2}, Nothing},
        Z <: Union{Array{Float64, 2}, Nothing},
        P <: Union{Array{Float64, 2}, Nothing},
        LASSO <: Union{Array{Bool,2}, Nothing},
        LASSO_PEN <: Union{Float64, Nothing},
        RIDGE <: Union{Array{Bool,2}, Nothing},
        RIDGE_PEN <: Union{Float64, Nothing}}
    ram::RAM
    data::DATA
    par::PAR
    mstruc::MSTRUC
    logl::LOGL
    opt::OPT
    est::EST
    obs_cov::OBS_COV
    imp_cov::IMP_COV
    obs_mean::OBS_MEAN
    opt_result::OPT_RESULT
    se::SE
    z::Z
    p::P
    lasso::LASSO
    lasso_pen::LASSO_PEN
    ridge::RIDGE
    ridge_pen::RIDGE_PEN
    model{RAM, DATA, PAR, MSTRUC, LOGL, OPT, EST, OBS_COV, IMP_COV, OBS_MEAN, OPT_RESULT, SE, Z, P, LASSO, LASSO_PEN, RIDGE, RIDGE_PEN}(
            ram, data, par,
            mstruc,
            logl,
            opt,
            est,
            obs_cov,
            imp_cov,
            obs_mean,
            opt_result,
            se,
            z,
            p,
            lasso,
            lasso_pen,
            ridge,
            ridge_pen) where {
                    RAM <: Function,
                    DATA <: Union{Matrix{Float64}, Nothing},
                    PAR <: Union{Array{Float64, 1}, Nothing},
                    MSTRUC <: Union{Bool, Nothing},
                    LOGL <: Union{Float64, Nothing},
                    OPT <: Union{String, Nothing},
                    EST <: Union{Function, Nothing},
                    OBS_COV <: Union{Matrix{Float64}, Nothing},
                    IMP_COV <: Union{Matrix{Float64}, Nothing},
                    OBS_MEAN <: Union{Array{Float64, 2}, Nothing},
                    OPT_RESULT <: Any,
                    SE <: Union{Array{Float64, 2}, Nothing},
                    Z <: Union{Array{Float64, 2}, Nothing},
                    P <: Union{Array{Float64, 2}, Nothing},
                    LASSO <: Union{Array{Bool,2}, Nothing},
                    LASSO_PEN <: Union{Float64, Nothing},
                    RIDGE <: Union{Array{Bool,2}, Nothing},
                    RIDGE_PEN <: Union{Float64, Nothing}} =
    new(ram, data, par,
            mstruc,
            logl,
            opt,
            est,
            obs_cov,
            imp_cov,
            obs_mean,
            opt_result,
            se,
            z,
            p,
            lasso,
            lasso_pen,
            ridge,
            ridge_pen)
end

model(ram::RAM, data::DATA, par::PAR;
        mstruc::MSTRUC = false,
        logl::LOGL = nothing,
        opt::OPT = "LBFGS",
        est::EST = nothing,
        obs_cov::OBS_COV = nothing,
        imp_cov::IMP_COV = nothing,
        obs_mean::OBS_MEAN = nothing,
        opt_result::OPT_RESULT = nothing,
        se::SE = nothing,
        z::Z = nothing,
        p::P = nothing,
        lasso::LASSO = nothing,
        lasso_pen::LASSO_PEN = nothing,
        ridge::RIDGE = nothing,
        ridge_pen::RIDGE_PEN = nothing) where {
                RAM <: Function,
                DATA <: Union{Matrix{Float64}, Nothing},
                PAR <: Union{Array{Float64, 1}, Nothing},
                MSTRUC <: Union{Bool, Nothing},
                LOGL <: Union{Float64, Nothing},
                OPT <: Union{String, Nothing},
                EST <: Union{Function, Nothing},
                OBS_COV <: Union{Matrix{Float64}, Nothing},
                IMP_COV <: Union{Matrix{Float64}, Nothing},
                OBS_MEAN <: Union{Array{Float64, 2}, Nothing},
                OPT_RESULT <: Any,
                SE <: Union{Array{Float64, 2}, Nothing},
                Z <: Union{Array{Float64, 2}, Nothing},
                P <: Union{Array{Float64, 2}, Nothing},
                LASSO <: Union{Array{Bool,2}, Nothing},
                LASSO_PEN <: Union{Float64, Nothing},
                RIDGE <: Union{Array{Bool,2}, Nothing},
                RIDGE_PEN <: Union{Float64, Nothing}} =
        model{RAM, DATA, PAR, MSTRUC, LOGL, OPT, EST, OBS_COV, IMP_COV, OBS_MEAN, OPT_RESULT, SE, Z, P, LASSO, LASSO_PEN, RIDGE, RIDGE_PEN}(
                ram, data, par,
                mstruc,
                logl,
                opt,
                est,
                obs_cov,
                imp_cov,
                obs_mean,
                opt_result,
                se,
                z,
                p,
                lasso,
                lasso_pen,
                ridge,
                ridge_pen)

struct ram{T}
        S::T
        F::T
        A::T
end

mutable struct teststruc{
    A <: Union{Float64, Nothing},
    B <: Union{Float64, Nothing}}
    a::A
    b::B
    teststruc{A, B}(a, b) where {
        A <: Union{Float64, Nothing},
        B <: Union{Float64, Nothing}} =
    new(a, b)
end

teststruc(a::A, b::B) where {
    A <: Union{Float64, Nothing},
    B <: Union{Float64, Nothing}} =
    teststruc{A, B}(a, b)

tf1 = teststruc(3.0, 4.0)
tf2 = teststruc(3.0, nothing)
tn2 = teststruc(nothing, 3.0)
tn1 = teststruc(nothing, nothing)

function func(obj::teststruc{Float64; b::Float64})
    obj.a
end

function func(obj::teststruc{Nothing, Union{Float64, Nothing}})
    tf.b
end

func(tf2)
