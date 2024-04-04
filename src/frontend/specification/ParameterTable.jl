############################################################################################
### Types
############################################################################################

mutable struct ParameterTable{C, V} <: AbstractParameterTable
    columns::C
    variables::V
    params::Vector{Symbol}
end

############################################################################################
### Constructors
############################################################################################

# constuct an empty table
function ParameterTable(; observed_vars::Union{AbstractVector{Symbol}, Nothing}=nothing,
                          latent_vars::Union{AbstractVector{Symbol}, Nothing}=nothing,
                          params::Union{AbstractVector{Symbol}, Nothing}=nothing)
    columns = (
        from = Vector{Symbol}(),
        relation = Vector{Symbol}(),
        to = Vector{Symbol}(),
        free = Vector{Bool}(),
        value_fixed = Vector{Float64}(),
        start = Vector{Float64}(),
        estimate = Vector{Float64}(),
        se = Vector{Float64}(),
        param = Vector{Symbol}(),
    )

    vars = (
        latent = !isnothing(latent_vars) ? copy(latent_vars) : Vector{Symbol}(),
        observed = !isnothing(observed_vars) ? copy(observed_vars) : Vector{Symbol}(),
        sorted = Vector{Symbol}()
    )

    return ParameterTable(columns, vars,
                          !isnothing(params) ? copy(params) : Vector{Symbol}())
end

# checks if params vector contain all ids from param_refs
# and optionally appends missing ones
function check_params(params::AbstractVector{Symbol},
                      param_refs::AbstractVector{Symbol};
                      append::Bool = false)
    param_refset = Set(id for id in param_refs if id != :const)
    undecl_params = setdiff(param_refset, params)
    if !isempty(undecl_params)
        if append # append preserving the order
            appended_params = Set{Symbol}()
            for param in param_refs
                if (param ∈ undecl_params) && (param ∉ appended_params)
                    push!(params, param)
                    push!(appended_params, param)
                end
            end
        else
            throw(ArgumentError("$(length(undecl_params)) parameters present in the table, but are not declared: " *
                            join(sort!(collect(undecl_params)))))
        end
    end
end

# new parameter table with different parameters order
function ParameterTable(partable::ParameterTable;
                        params::Union{AbstractVector{Symbol}, Nothing}=nothing)
    isnothing(params) || check_params(params, partable.columns.param)

    newtable = ParameterTable(observed_vars = observed_vars(partable),
                              latent_vars = latent_vars(partable),
                              params = params)
    newtable.columns = NamedTuple(col => copy(values)
                                  for (col, values) in pairs(partable.columns))

    return newtable
end

vars(partable::ParameterTable) =
    !isempty(partable.variables.sorted) ? partable.variables.sorted :
    vcat(partable.variables.observed, partable.variables.latent)
observed_vars(partable::ParameterTable) = partable.variables.observed
latent_vars(partable::ParameterTable) = partable.variables.latent

nvars(partable::ParameterTable) =
    length(partable.variables.latent) + length(partable.variables.observed)

############################################################################################
### Convert to other types
############################################################################################

function Base.convert(::Type{Dict}, partable::ParameterTable)
    return partable.columns
end

function Base.convert(::Type{ParameterTable}, partable::ParameterTable;
                      params::Union{AbstractVector{Symbol}, Nothing}=nothing)
    return isnothing(params) || partable.params == params ? partable :
        ParameterTable(partable; params=params)
end

function DataFrames.DataFrame(
        partable::ParameterTable;
        columns::Union{AbstractVector{Symbol}, Nothing} = nothing)
    if isnothing(columns)
        columns = [col for (col, vals) in pairs(partable.columns)
                   if length(vals) > 0]
    end
    return DataFrame([col => partable.columns[col]
                      for col in columns])
end

############################################################################################
### Pretty Printing
############################################################################################

function Base.show(io::IO, partable::ParameterTable)
    relevant_columns = [
        :from,
        :relation,
        :to,
        :free,
        :value_fixed,
        :start,
        :estimate,
        :se,
        :param]
    shown_columns = filter!(col -> haskey(partable.columns, col) && length(partable.columns[col]) > 0,
                            relevant_columns)

    as_matrix = mapreduce(col -> partable.columns[col], hcat, shown_columns)
    pretty_table(
        io,
        as_matrix,
        header = (
            shown_columns,
            [eltype(partable.columns[col]) for col in shown_columns]
        ),
        tf = PrettyTables.tf_compact)

    print(io, "Latent Variables:    $(partable.variables.latent) \n")
    print(io, "Observed Variables:  $(partable.variables.observed) \n")
end

############################################################################################
### Additional Methods
############################################################################################

# Iteration --------------------------------------------------------------------------------
ParameterTableRow = @NamedTuple begin
    from::Symbol
    relation::Symbol
    to::Symbol
    free::Bool
    value_fixed::Any
    param::Symbol
end

Base.getindex(partable::ParameterTable, i::Integer) =
    (from = partable.columns.from[i],
     relation = partable.columns.relation[i],
     to = partable.columns.to[i],
     free = partable.columns.free[i],
     value_fixed = partable.columns.value_fixed[i],
     param = partable.columns.param[i],
    )

Base.length(partable::ParameterTable) = length(first(partable.columns))
Base.eachindex(partable::ParameterTable) = Base.OneTo(length(partable))

Base.eltype(::Type{<:ParameterTable}) = ParameterTableRow
Base.iterate(partable::ParameterTable) = iterate(partable, 1)
Base.iterate(partable::ParameterTable, i::Integer) = i > length(partable) ? nothing : (partable[i], i + 1)

# Sorting ----------------------------------------------------------------------------------

struct CyclicModelError <: Exception
    msg::AbstractString
end

Base.showerror(io::IO, e::CyclicModelError) = print(io, e.msg)

"""
    sort_vars!(partable::ParameterTable)
    sort_vars!(partables::EnsembleParameterTable)

Sort variables in `partable` so that all independent variables are
before the dependent variables and store it in `partable.variables.sorted`.

If the relations between the variables are acyclic, sorting will
make the resulting `A` matrix in the *RAM* model lower triangular
and allow faster calculations.
"""
function sort_vars!(partable::ParameterTable)

    vars = [partable.variables.latent;
            partable.variables.observed]

    # regression edges (excluding intercept)
    edges = [(from, to)
             for (reltype, from, to) in
                    zip(partable.columns.relation,
                        partable.columns.from,
                        partable.columns.to)
             if (reltype == :→) && (from != Symbol("1"))]
    sort!(edges, by=last) # sort edges by target

    sorted_vars = Vector{Symbol}()

    while !isempty(vars)

        acyclic = false

        for (i, var) in enumerate(vars)
            # check if var has any incoming edge
            eix = searchsortedfirst(edges, (var, var), by=last)
            if !(eix <= length(edges) && last(edges[eix]) == var)
                # var is source, no edges to it
                push!(sorted_vars, var)
                deleteat!(vars, i)
                # remove var outgoing edges
                filter!(e -> e[1] != var, edges)
                acyclic = true
                break
            end
        end

        # if acyclic is false, all vars have incoming edge
        acyclic || throw(CyclicModelError("your model is cyclic and therefore can not be ordered"))
    end

    copyto!(resize!(partable.variables.sorted, length(sorted_vars)),
            sorted_vars)
    @assert length(partable.variables.sorted) == nvars(partable)

    return partable
end

"""
    sort_vars(partable::ParameterTable)
    sort_vars(partables::EnsembleParameterTable)

Sort variables in `partable` so that all independent variables are
before the dependent variables, and return the copy of `partable`.

See [sort_vars!](@ref) for in-place version.
"""
sort_vars(partable::ParameterTable) = sort_vars!(deepcopy(partable))

# add a row --------------------------------------------------------------------------------

function Base.push!(partable::ParameterTable, d::NamedTuple)
    for key in keys(d)
        push!(partable.columns[key], d[key])
    end
end

############################################################################################
### Update Partable from Fitted Model
############################################################################################

# update generic ---------------------------------------------------------------------------
function update_partable!(partable::ParameterTable,
                          column::Symbol,
                          param_values::AbstractDict{Symbol},
                          default::Any = nothing)
    coldata = partable.columns[column]
    resize!(coldata, length(partable))
    for (i, id) in enumerate(partable.columns.param)
        (id == :const) && continue
        if haskey(param_values, id)
            coldata[i] = param_values[id]
        else
            if isnothing(default)
                throw(KeyError(id))
            elseif (default isa AbstractVector) &&
               (length(default) == length(partable))
                coldata[i] = default[i]
            else
                coldata[i] = default
            end
        end
    end
    return partable
end

function update_partable!(partable::ParameterTable,
                          column::Symbol,
                          params::AbstractVector{Symbol},
                          values::AbstractVector,
                          default::Any = nothing)
    length(params) == length(values) ||
        throw(ArgumentError("The length of `params` ($(length(params))) and their `values` ($(length(values))) must be the same"))
    dup_params = nonunique(params)
    isempty(dup_params) || throw(ArgumentError("Duplicate parameters detected: $(join(dup_params, ", "))"))
    params_dict = Dict(zip(params, values))
    update_partable!(partable, column, params_dict, default)
end

# update estimates -------------------------------------------------------------------------
"""
    update_estimate!(
        partable::AbstractParameterTable,
        sem_fit::SemFit)

Write parameter estimates from `sem_fit` to the `:estimate` column of `partable`
"""
update_estimate!(partable::ParameterTable, sem_fit::SemFit) =
    update_partable!(partable, :estimate, params(sem_fit), sem_fit.solution,
                     partable.columns.value_fixed)

# fallback method for ensemble
update_estimate!(partable::AbstractParameterTable, sem_fit::SemFit) =
    update_partable!(partable, :estimate, params(sem_fit), sem_fit.solution)

# update starting values -------------------------------------------------------------------
"""
    update_start!(partable::AbstractParameterTable, sem_fit::SemFit)
    update_start!(partable::AbstractParameterTable, model::AbstractSem, start_val; kwargs...)

Write starting values from `sem_fit` or `start_val` to the `:estimate` column of `partable`.

# Arguments
- `start_val`: either a vector of starting values or a function to compute starting values
    from `model`
- `kwargs...`: are passed to `start_val`
"""
update_start!(partable::AbstractParameterTable, sem_fit::SemFit) =
    update_partable!(partable, :start, params(sem_fit), sem_fit.start_val,
                     partable.columns.value_fixed)

function update_start!(
        partable::AbstractParameterTable,
        model::AbstractSem,
        start_val;
        kwargs...)
    if !(start_val isa Vector)
        start_val = start_val(model; kwargs...)
    end
    return update_partable!(partable, :start, params(model), start_val)
end

# update partable standard errors ----------------------------------------------------------
"""
    update_se_hessian!(
        partable::AbstractParameterTable,
        fit::SemFit;
        method = :finitediff)

Write hessian standard errors computed for `sem_fit` to the `:se` column of `partable`

# Arguments
- `method::Symbol`: how to compute the hessian, see [se_hessian](@ref) for more information.

# Examples

"""
function update_se_hessian!(
        partable::AbstractParameterTable,
        fit::SemFit;
        method = :finitediff)
    se = se_hessian(fit; method = method)
    return update_partable!(partable, :se, params(fit), se)
end

function variance_params(partable::ParameterTable)
    res = [param for (param, rel, from, to) in
                zip(partable.columns.param, partable.columns.relation,
                    partable.columns.from, partable.columns.to)
           if (rel == :↔) && (from == to)]
    unique!(res)
end

"""
    param_values!(out::AbstractVector, partable::ParameterTable,
                  col::Symbol = :estimate)

Extract parameter values from the `col` column of `partable`
into the `out` vector.

The `out` vector should be of `nparams(partable)` length.
The *i*-th element of the `out` vector will contain the
value of the *i*-th parameter from `params(partable)`.

Note that the function combines the duplicate occurences of the
same parameter in `partable` and will raise an error if the
values do not match.
"""
function param_values!(out::AbstractVector, partable::ParameterTable,
                       col::Symbol = :estimate)
    (length(out) == nparams(partable)) ||
        throw(DimensionMismatch("The length of parameter values vector ($(length(out))) does not match the number of parameters ($(nparams(partable)))"))
    param_index = Dict(param => i for (i, param) in enumerate(params(partable)))
    param_values_col = partable.columns[col]
    for (i, param) in enumerate(partable.columns.param)
        (param == :const) && continue
        param_ind = get(param_index, param, nothing)
        @assert !isnothing(param_ind) "Parameter table contains unregistered parameter :$param at row #$i"
        val = param_values_col[i]
        if !isnan(out[param_ind])
            @assert out[param_ind] ≈ val atol=1E-10 "Parameter :$param value at row #$i ($val) differs from the earlier encountered value ($(out[param_ind]))"
        else
            out[param_ind] = val
        end
    end
    return out
end

"""
    param_values(out::AbstractVector, col::Symbol = :estimate)

Extract parameter values from the `col` column of `partable`.

Returns the values vector. The *i*-th element corresponds to
the value of *i*-th parameter from `params(partable)`.

Note that the function combines the duplicate occurences of the
same parameter in `partable` and will raise an error if the
values do not match.
"""
param_values(partable::ParameterTable, col::Symbol = :estimate) =
    param_values!(fill(NaN, nparams(partable)), partable, col)

"""
    lavaan_param_values!(out::AbstractVector, partable_lav,
                         partable::ParameterTable,
                         lav_col::Symbol = :est, lav_group = nothing)

Extract parameter values from the `partable_lav` lavaan model that
match the parameters of `partable` into the `out` vector.

The method sets the *i*-th element of the `out` vector to
the value of *i*-th parameter from `params(partable)`.

Note that the lavaan and `partable` models are matched by the
the names of variables in the tables (`from` and `to` columns)
as well as the type of their relationship (`relation` column),
and not by the names of the model parameters.
"""
function lavaan_param_values!(out::AbstractVector,
                              partable_lav, partable::ParameterTable,
                              lav_col::Symbol = :est, lav_group = nothing)

    # find indices of all df row where f is true
    findallrows(f::Function, df) = findall(f(r) for r in eachrow(df))

    (length(out) == nparams(partable)) || throw(DimensionMismatch("The length of parameter values vector ($(length(out))) does not match the number of parameters ($(nparams(partable)))"))
    partable_mask = findall(partable.columns[:free])
    param_index = Dict(param => i for (i, param) in enumerate(params(partable)))

    lav_values = partable_lav[:, lav_col]
    for (from, to, type, id) in
        zip([view(partable.columns[k], partable_mask)
             for k in [:from, :to, :relation, :param]]...)

        lav_ind = nothing

        if from == Symbol("1")
            lav_ind = findallrows(r -> r[:lhs] == String(to) && r[:op] == "~1" &&
                                  (isnothing(lav_group) || r[:group] == lav_group), partable_lav)
        else
            if type == :↔
                lav_type = "~~"
            elseif type == :→
                if (from ∈ partable.variables.latent) && (to ∈ partable.variables.observed)
                    lav_type = "=~"
                else
                    lav_type = "~"
                    from, to = to, from
                end
            end

            if lav_type == "~~"
                lav_ind = findallrows(r -> ((r[:lhs] == String(from) && r[:rhs] == String(to)) ||
                                        (r[:lhs] == String(to) && r[:rhs] == String(from))) &&
                                        r[:op] == lav_type &&
                                        (isnothing(lav_group) || r[:group] == lav_group),
                                      partable_lav)
            else
                lav_ind = findallrows(r -> r[:lhs] == String(from) && r[:rhs] == String(to) && r[:op] == lav_type &&
                                      (isnothing(lav_group) || r[:group] == lav_group),
                                      partable_lav)
            end
        end

        if length(lav_ind) == 0
            throw(ErrorException("Parameter $id ($from $type $to) could not be found in the lavaan solution"))
        elseif length(lav_ind) > 1
            throw(ErrorException("At least one parameter was found twice in the lavaan solution"))
        end

        param_ind = param_index[id]
        param_val = lav_values[lav_ind[1]]
        if isnan(out[param_ind])
            out[param_ind] = param_val
        else
            @assert out[param_ind] ≈ param_val atol=1E-10 "Parameter :$id value at row #$lav_ind ($param_val) differs from the earlier encountered value ($(out[param_ind]))"
        end
    end

    return out
end

"""
    lavaan_param_values(partable_lav, partable::ParameterTable,
                        lav_col::Symbol = :est, lav_group = nothing)

Extract parameter values from the `partable_lav` lavaan model that
match the parameters of `partable`.

The `out` vector should be of `nparams(partable)` length.
The *i*-th element of the `out` vector will contain the
value of the *i*-th parameter from `params(partable)`.

Note that the lavaan and `partable` models are matched by the
the names of variables in the tables (`from` and `to` columns),
and the type of their relationship (`relation` column),
but not by the ids of the model parameters.
"""
lavaan_param_values(partable_lav, partable::ParameterTable,
                    lav_col::Symbol = :est, lav_group = nothing) =
    lavaan_param_values!(fill(NaN, nparams(partable)),
                         partable_lav, partable, lav_col, lav_group)
