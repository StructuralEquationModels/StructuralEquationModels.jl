# Hessians and symbolic precomputation

- `∇²Σ(::RAMSymbolic)` -> pre-allocated array for ``∂vec(Σ)/∂θᵀ``
- `∇²Σ_function(::RAMSymbolic)` -> function to overwrite `∇²Σ` in place
