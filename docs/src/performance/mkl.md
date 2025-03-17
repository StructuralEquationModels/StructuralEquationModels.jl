# Use MKL

Depending on the machine and the specific models, we sometimes observed large performance benefits from using [MKL](https://en.wikipedia.org/wiki/Math_Kernel_Library) as a backend for matrix operations. 
Fortunately, this is very simple to do in julia, so you can just try it out and check if turns out to be beneficial in your use case.

We install the [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) package:

```julia
using Pkg; Pkg.add("MKL")
```

Whenever we execute `using MKL` in a julia session, from now on MKL will be used as a backend.
To check the installation:

```julia
using LinearAlgebra

BLAS.get_config()

using MKL

BLAS.get_config()
```

To check the performance implications for fitting a SEM, you can use the [`BenchmarkTools`](https://github.com/JuliaCI/BenchmarkTools.jl) package:

```julia
using BenchmarkTools

@benchmark fit($your_model)

using MKL

@benchmark fit($your_model)
```