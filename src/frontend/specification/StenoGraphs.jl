## Interface to the StenoGraphs package

############################################################################################
### Define Modifiers
############################################################################################

AbstractStenoGraph = AbstractVector

# fixed parameter values
struct Fixed{N} <: EdgeModifier
    value::N
end
fixed(args...) = Fixed(args)
Fixed(value::Int) = Fixed(Float64(value))

# start values
struct Start{N} <: EdgeModifier
    value::N
end
start(args...) = Start(args)
Start(value::Int) = Start(Float64(value))

# labels for equality constraints
struct Label{N} <: EdgeModifier
    value::N
end
label(args...) = Label(args)

############################################################################################
### constructor for parameter table from graph
############################################################################################

function ParameterTable(graph::AbstractStenoGraph;
                        observed_vars, latent_vars,
                        params::Union{AbstractVector{Symbol}, Nothing} = nothing,
                        group::Integer = 1, param_prefix::Symbol = :θ)
    graph = unique(graph)
    n = length(graph)

    partable = ParameterTable(
        latent_vars = latent_vars,
        observed_vars = observed_vars,
        params = params)
    from = resize!(partable.columns.from, n)
    parameter_type = resize!(partable.columns.parameter_type, n)
    to = resize!(partable.columns.to, n)
    free = fill!(resize!(partable.columns.free, n), true)
    value_fixed = fill!(resize!(partable.columns.value_fixed, n), NaN)
    start = fill!(resize!(partable.columns.start, n), NaN)
    param = fill!(resize!(partable.columns.param, n), Symbol(""))
    # group = Vector{Symbol}(undef, n)
    # start_partable = zeros(Bool, n)

    for (i, element) in enumerate(graph)
        edge = element isa ModifiedEdge ? element.edge : element
        from[i] = edge.src.node
        to[i] = edge.dst.node
        if edge isa DirectedEdge
            parameter_type[i] = :→
        elseif edge isa UndirectedEdge
            parameter_type[i] = :↔
        else
            throw(ArgumentError("The graph contains an unsupported edge of type $(typeof(edge))."))
        end
        if element isa ModifiedEdge
            for modifier in values(element.modifiers)
                modval = modifier.value[group]
                if modifier isa Fixed
                    if modval == :NaN
                        free[i] = true
                        value_fixed[i] = 0.0
                    else
                        free[i] = false
                        value_fixed[i] = modval
                    end
                elseif modifier isa Start
                    start_partable[i] = modval == :NaN
                    start[i] = modval
                elseif modifier isa Label
                    if modval == :NaN
                        throw(DomainError(NaN, "NaN is not allowed as a parameter label."))
                    end
                    param[i] = modval
                end
            end
        end
    end

    # assign identifiers for parameters that are not labeled
    current_id = 1
    for i in 1:length(param)
        if param[i] == Symbol("")
            if free[i]
                param[i] = Symbol(param_prefix, :_, current_id)
                current_id += 1
            else
                param[i] = :const
            end
        elseif !free[i]
            @warn "You labeled a constant ($(param[i])=$(value_fixed[i])). Please check if the labels of your graph are correct."
        end
    end

    if isnothing(params)
        # collect the unique params in the order of appearance
        @assert isempty(partable.params)
        ids = Set{Symbol}()
        for id in param
            if (id != :const) && (id ∉ ids)
                push!(ids, id)
                push!(partable.params, id)
            end
        end
    end
    check_params(partable.params, param)

    return partable
end

############################################################################################
### constructor for EnsembleParameterTable from graph
############################################################################################

function EnsembleParameterTable(graph::AbstractStenoGraph;
                                observed_vars, latent_vars, groups,
                                params::Union{AbstractVector{Symbol}, Nothing} = nothing)

    graph = unique(graph)

    partables = Dict(group => ParameterTable(
            graph;
            observed_vars = observed_vars,
            latent_vars = latent_vars,
            params = params,
            group = i,
            param_prefix = Symbol(:g, group))
            for (i, group) in enumerate(groups))
    return EnsembleParameterTable(partables, params = params)
end