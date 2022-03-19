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
    label = Vector{Symbol}(undef, n); label .= Symbol("")
    start = zeros(n)
    estimate = zeros(n)
    identifier = Vector{Symbol}(undef, n)
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
                    label[i] = modifier.value
                end
            end
        end 
    end

    n_labels_unique = size(unique(label), 1) - 1
    n_labels = sum(.!(label .== Symbol("")))
    n_parameters = sum(free) - n_labels + n_labels_unique
    
    identifier = Symbol.(:θ_, 1:n_parameters)
    identifier_copy = copy(identifier)
    label_identifier = Dict{Symbol, Symbol}()
    identifier_long = Vector{Symbol}()

    for label in label[free]
        if label == Symbol("")
            push!(identifier_long, popfirst!(identifier_copy))
        else
            if haskey(label_identifier, label)
                push!(identifier_long, label_identifier[label])
            else
                push!(label_identifier, label => first(identifier_copy))
                push!(identifier_long, popfirst!(identifier_copy))
            end
        end
    end

    identifier_out = Vector{Symbol}(undef, length(to))
    identifier_out[.!free] .= :const
    identifier_out[free] .= identifier_long

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