## Interface to the StenoGraphs package

############################################################################
### Define Modifiers
############################################################################

# fixed parameter values
struct Fixed{N} <: EdgeModifier
    value::N
end
fixed(value) = Fixed(value)
Fixed(value::Int) = Fixed(Float64(value))

# start values
struct Start{N} <: EdgeModifier
    value::N
end
start(value) = Start(value)
Start(value::Int) = Start(Float64(value))

# labels for equality constraints
struct Label{N <: Symbol} <: EdgeModifier
    value::N
end
label(value) = Label(value)

############################################################################
### constructor for parameter table from graph
############################################################################

function ParameterTable(;graph, observed_vars, latent_vars)
    n = length(graph)
    from = Vector{Symbol}(undef, n)
    parameter_type = Vector{Symbol}(undef, n)
    to = Vector{Symbol}(undef, n)
    free = ones(Bool, n)
    value_fixed = zeros(n)
    start = zeros(n)
    estimate = zeros(n)
    identifier = Vector{Symbol}(undef, n); identifier .= Symbol("")
    # group = Vector{Symbol}(undef, n)
    # start_partable = zeros(Bool, n)

    sorted_vars = Vector{Symbol}()

    for (i, element) in enumerate(graph)
        if element isa DirectedEdge
            from[i] =  element.src.node
            to[i] =  element.dst.node
            parameter_type[i] = :→
        elseif element isa UndirectedEdge
            from[i] =  element.src.node
            to[i] =  element.dst.node
            parameter_type[i] = :↔
        elseif element isa ModifiedEdge
            if element.edge isa DirectedEdge
                from[i] =  element.edge.src.node
                to[i] =  element.edge.dst.node
                parameter_type[i] = :→
            elseif element.edge isa UndirectedEdge
                from[i] =  element.edge.src.node
                to[i] =  element.edge.dst.node
                parameter_type[i] = :↔
            end
            for modifier in values(element.modifiers)
                if modifier isa Fixed
                    free[i] = false
                    value_fixed[i] = modifier.value
                elseif modifier isa Start
                    start_partable[i] = true
                    start[i] = modifier.value
                elseif modifier isa Label
                    identifier[i] = modifier.value
                end
            end
        end 
    end

    # make identifiers for parameters that are not labeled
    current_id = 1
    for i in 1:length(identifier)
        if identifier[i] == Symbol("")
            identifier[i] = Symbol(:θ_, current_id)
        end
        current_id += 1
    end

    return StructuralEquationModels.ParameterTable(
        Dict(
            :from => from,
            :parameter_type => parameter_type,
            :to => to,
            :free => free,
            :value_fixed => value_fixed,
            :label => label,
            :start => start,
            :estimate => estimate,
            :identifier => identifier_out),
            #:group => ,
            #:start_partable => start_partable),
        Dict(
            :latent_vars => latent_vars,
            :observed_vars => observed_vars,
            :sorted_vars => sorted_vars)
    )
end