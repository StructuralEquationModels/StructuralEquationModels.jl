#####################################################################################################
# struct
#####################################################################################################

mutable struct SemFit{Mi, So, St, Mo, O}
    minimum::Mi
    solution::So
    start_val::St
    model::Mo
    optimization_result::O
end

##############################################################
# pretty printing
##############################################################

function Base.show(io::IO, semfit::SemFit)
    print(io, "Fitted Structural Equation Model \n")
    print(io, "================================ \n")
    print(io, "------------- Model ------------ \n")
    print(io, semfit.model)
    print(io, "\n")
    #print(io, "Objective value: $(round(semfit.minimum, digits = 4)) \n")
    print(io, "----- Optimization result ------ \n")
    print(io, semfit.optimization_result)
end