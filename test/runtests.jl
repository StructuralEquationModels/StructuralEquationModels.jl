using sem, Test, Feather

function holz_onef_mod(x)
    S = [x[1] 0 0 0
        0 x[2] 0 0
        0 0 x[3] 0
        0 0 0 x[4]]

    F = [1 0 0 0
        0 1 0 0
        0 0 1 0]

    A = [0 0 0 1
        0 0 0 x[5]
        0 0 0 x[6]
        0 0 0 0]

    return (S, F, A)
end

pwd()

holz_onef_dat = Feather.read("test/comparisons/holz_onef_dat.feather")
holz_onef_par = Feather.read("test/comparisons/holz_onef_par.feather")

@testset "test model spec" begin
    mymod = model(holz_onef_mod, holz_onef_dat, [0.5, 0.5, 0.5, 0.5, 1.0, 1.0])
    @test all([i in [
        :ram
        :par
        :data
        :opt
        :est
        :obs_cov
        :imp_cov
        :obs_mean
        :logl
        :opt_result
        :se
        :z
        :p] for i = keys(mymod)])
      @test mymod[:data] == convert(Matrix{Float64}, holz_onef_dat)
      @test mymod[:ram] == holz_onef_mod
    #sem.sem_obs_cov(mymod)

    #sem.sem_imp_cov(mymod)

    #sem_obs_mean(mymod)

    #sem_logl(mymod)

end



mymod_lbfgs =
    model(holz_onef_mod, holz_onef_dat, [0.5, 0.5, 0.5, 0.5, 1.0, 1.0])
sem_fit!(mymod_lbfgs)

mymod_newton =
    model(holz_onef_mod, holz_onef_dat, [0.5, 0.5, 0.5, 0.5, 1.0, 1.0],
    sem.ML, "Newton")
sem_fit!(mymod_newton)

@testset "holz_onef_par" begin
    # by position- change!
    lav_par = vcat(holz_onef_par.est[4:7], holz_onef_par.est[2:3])
# LBFGS
    par_diff_lbfgs = abs.(mymod_lbfgs[:par] .- lav_par)
    @test all(par_diff_lbfgs .< 0.01)
# Newton
    par_diff_newton = abs.(mymod_newton[:par] .- lav_par)
    @test all(par_diff_newton .< 0.01)
#

end

delta_method!(mymod_newton)
delta_method!(mymod_lbfgs)

@testset "holz_onef_delta_se_p_z" begin
    lav_se = vcat(holz_onef_par.se[4:7], holz_onef_par.se[2:3])
    lav_z = vcat(holz_onef_par.z[4:7], holz_onef_par.z[2:3])
    lav_p = vcat(holz_onef_par.p[4:7], holz_onef_par.p[2:3])

    se_diff_newton = abs.(mymod_newton[:se] .- lav_se)
    se_diff_lbfgs = abs.(mymod_lbfgs[:se] .- lav_se)
    z_diff_newton = abs.(mymod_newton[:z] .- lav_z)
    z_diff_lbfgs = abs.(mymod_lbfgs[:z] .- lav_z)
    p_diff_newton = abs.(mymod_newton[:p] .- lav_p)
    p_diff_lbfgs = abs.(mymod_lbfgs[:p] .- lav_p)

    # figure out appropriate error margin!
    # should be relative to absolute size
    @test all(se_diff_newton .< 0.01)
    @test all(se_diff_lbfgs .< 0.01)
    @test all(z_diff_newton .< 0.1)
    @test all(z_diff_lbfgs .< 0.1)
    @test all(p_diff_newton .< 0.01)
    @test all(p_diff_lbfgs .< 0.01)

end
