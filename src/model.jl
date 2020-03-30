mutable struct model
    ram::Union{Function, Nothing}
    data::Union{Matrix{Float64}, Nothing}
    par::Union{Array{Float64, 1}, Nothing}
    mstruc::Union{Bool, Nothing}
    logl::Union{Float64, Nothing}
    penalty::Union{Float64, Nothing}
    opt::Union{String, Nothing}
    est::Union{Function, Nothing}
    obs_cov::Union{Matrix{Float64}, Nothing}
    imp_cov::Union{Matrix{Float64}, Nothing}
    obs_mean::Union{Array{Float64, 2}, Nothing}
    opt_result::Any
    se::Union{Array{Float64, 2}, Nothing}
    z::Union{Array{Float64, 2}, Nothing}
    p::Union{Array{Float64, 2}, Nothing}
    reg::Union{String, Nothing}
    reg_vec::Union{Array{Bool,2}, Nothing}
    model(ram, data, par;
            mstruc = false,
            logl = nothing,
            penalty = nothing,
            opt = "LBFGS",
            est = nothing,
            obs_cov = nothing,
            imp_cov = nothing,
            obs_mean = nothing,
            opt_result = nothing,
            se = nothing,
            z = nothing,
            p = nothing,
            reg = nothing,
            reg_vec = nothing) =
    new(ram, convert(Matrix{Float64}, data), par,
            mstruc,
            logl,
            penalty,
            opt,
            est,
            obs_cov,
            imp_cov,
            obs_mean,
            opt_result,
            se,
            z,
            p,
            reg,
            reg_vec)
end
