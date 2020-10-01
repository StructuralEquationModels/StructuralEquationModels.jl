
function myloss(
    par,
    model,
    mult
    )
    #imp_det = det(model.imply.imp_cov)
    if !isposdef(model.imply.imp_cov)
        F = Inf
    else
        a = cholesky!(model.imply.imp_cov)
        ld = logdet(a)
        model.imply.imp_cov .= LinearAlgebra.inv!(a)
        mul!(mult, model.observed.obs_cov, model.imply.imp_cov)
        #mul!()
        F = ld +
            tr(mult)# -
            #semml.logdet_obs -
            #model.observed.n_man
    end
    return F
end

mult = zeros(11,11)
inverse = zeros(11,11)

model.imply(solution.minimizer)

mat = copy(model.imply.imp_cov)

model.imply.imp_cov .= copy(mat)

@btime myloss(solution.minimizer, model, mult, inverse)

a = 11 + model.loss.functions[1].logdet_obs

mat2 = copy(mat)

mat3 = cholesky!(mat2)

function myfun(mat3, res)
    res .= LinearAlgebra.inv!(mat3)
end

res = zeros(11,11)

@time myfun(mat3, res)

c = rand(11,11); d = rand(11,11)

mul!(c, d, mat3)

mat3
