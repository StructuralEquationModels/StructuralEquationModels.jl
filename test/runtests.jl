using sem, Test, Feather

function holz_onef_mod(x)
      S =   [x[1] 0 0 0
            0 x[2] 0 0
            0 0 x[3] 0
            0 0 0 x[4]]

      F =  [1 0 0 0
            0 1 0 0
            0 0 1 0]

      A =  [0 0 0 1
            0 0 0 x[5]
            0 0 0 x[6]
            0 0 0 0]

      return (S, F, A)
end

holz_onef_dat = Feather.read("test/comparisons/holz_onef_dat.feather")


@testset "test model spec" begin
    mymod = sem.model(holz_onef_mod, holz_onef_dat, [0.5, 0.5, 0.5, 0.5, 1.0, 1.0])
    @test all([i in [
            :ram
            :par
            :data
            :optim
            :obs_cov
            :imp_cov
            :obs_mean
            :logl
            :opt_result] for i = keys(mymod)])
      @test mymod[:data] == convert(Matrix{Float64}, holz_onef_dat)
      @test mymod[:ram] == holz_onef_mod
    #sem.sem_obs_cov(mymod)

    #sem.sem_imp_cov(mymod)

    #sem_obs_mean(mymod)

    #sem_logl(mymod)

end
