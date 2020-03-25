using sem, Test#, Feather

## Test one factor model with Holzinger_Swineford data

# import results from lavaan
#holz_onef_par = Feather.read("test/comparisons/holz_onef_par.feather")
#holz_onef_dat = Feather.read("test/comparisons/holz_onef_dat.feather")

#function holz_onef_mod(x)
#      S =   [x[1] 0 0 0
#            0 x[2] 0 0
#            0 0 x[3] 0
#            0 0 0 x[4]]
#
#      F =  [1 0 0 0
#            0 1 0 0
#            0 0 1 0]
#
#      A =  [0 0 0 1
#            0 0 0 x[5]
#            0 0 0 x[6]
#            0 0 0 0]
#
#      return (S, F, A)
#end


#x0 = append!([0.5, 0.5, 0.5, 0.5], ones(2))

#holz_onef_fit_lb = fit_sem(holz_onef_mod, holz_onef_dat, x0, ML, "LBFGS")

#holz_onef_fit_ne = fit_sem(holz_onef_mod, holz_onef_dat, x0, ML, "Newton")

@testset "sem.jl" begin
    #@test x0 == [0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
end
