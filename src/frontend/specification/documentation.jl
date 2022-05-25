"""
    ParameterTable(;graph, observed_vars, latent_vars, ...)
    ParameterTable(ram_matrices)

Return a `ParameterTable` constructed from `graph` or `ram_matrices`.

# Arguments
- `graph`: graph defined via `@StenoGraph`
- `observed_vars::Vector{Symbol}`: observed variable names
- `latent_vars::Vector{Symbol}`: latent variable names
- `parname::Symbol = :θ`: prefix for automatically generated parameter labels
- `ram_matrices::RAMMatrices`: a `RAMMatrices` object
    
# Examples
"""
function ParameterTable end

"""
    EnsembleParameterTable(;graph, observed_vars, latent_vars, groups)
    EnsembleParameterTable(args...; groups)

Return an `EnsembleParameterTable` constructed from `graph` or multiple `RAMMatrices`.

# Arguments
- `graph`: graph defined via `@StenoGraph`
- `observed_vars::Vector{Symbol}`: observed variable names
- `latent_vars::Vector{Symbol}`: latent variable names
- `groups::Vector{Symbol}`: group names
- `args...`: `RAMMatrices` for each model

# Examples
See the online documentation on [Multigroup models](@ref).
"""
function EnsembleParameterTable end


"""
    RAMMatrices(partable::ParameterTable)
    RAMMatrices(;A, S, F, M = nothing, parameters, colnames)
    RAMMatrices(partable::EnsembleParameterTable)
    
Return `RAMMatrices` constructed from a parameter table or individual matrices, 
or a dictionary with group names (Symbols) as keys and `RAMMatrices` as values from 
an `EnsembleParameterTable`.

# Arguments
- `partable`
- `A`: matrix of directed effects
- `S`: matrix of undirected effects
- `F`: filter matrix
- `M`: vector of mean effects
- `parameters::Vector{Symbol}`: parameter labels
- `colnames::Vector{Symbol}`: variable names corresponding to the matrix columns

# Examples
See the online documentation on XXX.
"""
function RAMMatrices end

"""
    fixed(args...)

Fix parameters to a certain value. For ensemble models, multiple values 
(one for each submodel/group) are needed.

# Examples
```julia
graph = @StenoGraph begin
    x → fixed(1)*y
end
```
"""
function fixed end

"""
    start(args...)

Define starting values for parameters. For ensemble models, multiple values 
(one for each submodel/group) are needed.

# Examples
```julia
graph = @StenoGraph begin
    x → start(1)*y
end
```
"""
function start end

"""
    label(args...)

Label parameters.
For ensemble models, multiple values (one for each submodel/group) are needed.

# Examples
```julia
graph = @StenoGraph begin
    x → label(:a)*y
end
```
"""
function label end