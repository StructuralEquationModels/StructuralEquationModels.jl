# class of constant loss functions for scaling 
# (e.g. to compare with other sem software)

struct SemConstant{C} <: LossFunction
    constant::C
end

function (constant::SemConstant)(par, model)
    return constant.constant
end
