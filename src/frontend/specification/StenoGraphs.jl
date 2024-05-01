## Interface to the StenoGraphs package

############################################################################################
### Define Modifiers
############################################################################################

#FIXME: remove when StenoGraphs.jl will provide AbstractStenoGraph
const AbstractStenoGraph = AbstractArray{T, 1} where {T <: StenoGraphs.AbstractEdge}

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

function ParameterTable(
    graph::AbstractStenoGraph;
    observed_vars,
    latent_vars,
    group::Integer = 1,
    param_prefix = :θ,
)
    graph = unique(graph)
    n = length(graph)

    columns = empty_partable_columns(n)
    from = columns[:from]
    parameter_type = columns[:parameter_type]
    to = columns[:to]
    free = columns[:free]
    value_fixed = columns[:value_fixed]
    start = columns[:start]
    param_refs = columns[:param]
    # group = Vector{Symbol}(undef, n)

    for (i, element) in enumerate(graph)
        edge = element isa ModifiedEdge ? element.edge : element
        from[i] = edge.src.node
        to[i] = edge.dst.node
        if edge isa DirectedEdge
            parameter_type[i] = :→
        elseif edge isa UndirectedEdge
            parameter_type[i] = :↔
        else
            throw(
                ArgumentError(
                    "The graph contains an unsupported edge of type $(typeof(edge)).",
                ),
            )
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
                    start[i] = modval
                elseif modifier isa Label
                    if modval == :NaN
                        throw(DomainError(NaN, "NaN is not allowed as a parameter label."))
                    end
                    param_refs[i] = modval
                end
            end
        end
    end

    # make identifiers for parameters that are not labeled
    current_id = 1
    for i in eachindex(param_refs)
        if param_refs[i] == Symbol("")
            if free[i]
                param_refs[i] = Symbol(param_prefix, :_, current_id)
                current_id += 1
            else
                param_refs[i] = :const
            end
        elseif !free[i]
            @warn "You labeled a constant ($(param_refs[i])=$(value_fixed[i])). Please check if the labels of your graph are correct."
        end
    end

    return ParameterTable(columns; latent_vars, observed_vars)
end

############################################################################################
### constructor for EnsembleParameterTable from graph
############################################################################################

function EnsembleParameterTable(
    graph::AbstractStenoGraph;
    observed_vars,
    latent_vars,
    groups,
)
    graph = unique(graph)

    partables = Dict(
        group => ParameterTable(
            graph;
            observed_vars = observed_vars,
            latent_vars = latent_vars,
            group = i,
            param_prefix = Symbol(:g, group),
        ) for (i, group) in enumerate(groups)
    )
    return EnsembleParameterTable(partables)
end
