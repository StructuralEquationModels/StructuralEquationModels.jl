## Interface to the StenoGraphs package

############################################################################################
### Define Modifiers
############################################################################################

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

function ParameterTable(; graph, observed_vars, latent_vars, g = 1, parname = :θ)
    graph = unique(graph)
    n = length(graph)
    from = Vector{Symbol}(undef, n)
    parameter_type = Vector{Symbol}(undef, n)
    to = Vector{Symbol}(undef, n)
    free = ones(Bool, n)
    value_fixed = zeros(n)
    start = zeros(n)
    estimate = zeros(n)
    params = Vector{Symbol}(undef, n)
    params .= Symbol("")
    # group = Vector{Symbol}(undef, n)
    # start_partable = zeros(Bool, n)

    sorted_vars = Vector{Symbol}()

    for (i, element) in enumerate(graph)
        if element isa DirectedEdge
            from[i] = element.src.node
            to[i] = element.dst.node
            parameter_type[i] = :→
        elseif element isa UndirectedEdge
            from[i] = element.src.node
            to[i] = element.dst.node
            parameter_type[i] = :↔
        elseif element isa ModifiedEdge
            if element.edge isa DirectedEdge
                from[i] = element.edge.src.node
                to[i] = element.edge.dst.node
                parameter_type[i] = :→
            elseif element.edge isa UndirectedEdge
                from[i] = element.edge.src.node
                to[i] = element.edge.dst.node
                parameter_type[i] = :↔
            end
            for modifier in values(element.modifiers)
                if modifier isa Fixed
                    if modifier.value[g] == :NaN
                        free[i] = true
                        value_fixed[i] = 0.0
                    else
                        free[i] = false
                        value_fixed[i] = modifier.value[g]
                    end
                elseif modifier isa Start
                    start_partable[i] = modifier.value[g] == :NaN
                    start[i] = modifier.value[g]
                elseif modifier isa Label
                    if modifier.value[g] == :NaN
                        throw(DomainError(NaN, "NaN is not allowed as a parameter label."))
                    end
                    params[i] = modifier.value[g]
                end
            end
        end
    end

    # make identifiers for parameters that are not labeled
    current_id = 1
    for i in 1:length(params)
        if (params[i] == Symbol("")) & free[i]
            params[i] = Symbol(parname, :_, current_id)
            current_id += 1
        elseif (params[i] == Symbol("")) & !free[i]
            params[i] = :const
        elseif (params[i] != Symbol("")) & !free[i]
            @warn "You labeled a constant. Please check if the labels of your graph are correct."
        end
    end

    return StructuralEquationModels.ParameterTable(
        Dict(
            :from => from,
            :parameter_type => parameter_type,
            :to => to,
            :free => free,
            :value_fixed => value_fixed,
            :start => start,
            :estimate => estimate,
            :param => params,
        ),
        Dict(
            :latent_vars => latent_vars,
            :observed_vars => observed_vars,
            :sorted_vars => sorted_vars,
        ),
    )
end

############################################################################################
### constructor for EnsembleParameterTable from graph
############################################################################################

function EnsembleParameterTable(; graph, observed_vars, latent_vars, groups)
    graph = unique(graph)

    partable = EnsembleParameterTable(nothing)

    for (i, group) in enumerate(groups)
        push!(
            partable.tables,
            Symbol(group) => ParameterTable(;
                graph = graph,
                observed_vars = observed_vars,
                latent_vars = latent_vars,
                g = i,
                parname = Symbol(:g, i),
            ),
        )
    end

    return partable
end
