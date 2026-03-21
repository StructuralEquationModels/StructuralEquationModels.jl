losstype(::Type{<:LossTerm{L, W, I}}) where {L, W, I} = L
losstype(term::LossTerm) = losstype(typeof(term))
loss(term::LossTerm) = term.loss
weight(term::LossTerm) = term.weight
id(term::LossTerm) = term.id

"""
    issemloss(term::LossTerm) -> Bool

Check if a SEM model term is a SEM loss function ([`SemLoss`](@ref)).
"""
issemloss(term::LossTerm) = isa(loss(term), SemLoss)

for f in (
    :implied,
    :observed,
    :nsamples,
    :observed_vars,
    :nobserved_vars,
    :vars,
    :nvars,
    :latent_vars,
    :nlatent_vars,
    :params,
    :nparams,
)
    @eval $f(term::LossTerm) = $f(loss(term))
end

function Base.show(io::IO, term::LossTerm)
    if !isnothing(id(term))
        print(io, ":$(id(term)): ")
    end
    print(io, nameof(losstype(term)))
    if issemloss(term)
        print(
            io,
            " ($(nsamples(term)) samples, $(nobserved_vars(term)) observed, $(nlatent_vars(term)) latent variables)",
        )
    end
    if !isnothing(weight(term))
        print(io, " w=$(round(weight(term), digits=3))")
    else
        print(io, " w=1")
    end
end

# scaling corrections for multigroup models

# fallback method for non-standard SemLoss type
multigroup_correction_scale(::Type{<:SemLoss}) = nothing

multigroup_correction_scale(::Type{<:SemFIML}) = 0
multigroup_correction_scale(::Type{<:SemML}) = 0
multigroup_correction_scale(::Type{<:SemWLS}) = -1

multigroup_correction_scale(loss::SemLoss) = multigroup_correction_scale(typeof(loss))

# calculate sem term weights for multigroup models
# correcting for the number of samples and the loss type
function multigroup_weights(semterms...)
    n = length(semterms)
    nsamples_total = sum(nsamples, semterms)
    semloss_type = check_same_semterm_type(semterms; throw_error = false)
    if isnothing(semloss_type)
        @info """
        Your ensemble model contains heterogeneous loss functions.
        Default weights of (#samples per group/#total samples) will be used
        """
        c = 0
    else
        c = multigroup_correction_scale(semloss_type)
        if isnothing(c)
            @info """
            We don't know how to choose group weights for the specified loss function.
            Default weights of (#samples per group/#total samples) will be used
            """
            c = 0
        end
    end
    return [(nsamples(term)+c) / (nsamples_total+n*c) for term in semterms]
end

############################################################################################
# constructor for Sem types
############################################################################################

function Sem(
    loss_terms...;
    params::Union{Vector{Symbol}, Nothing} = nothing,
    default_sem_weights = :nsamples,
)
    default_sem_weights ∈ [:nsamples, :uniform, :one] ||
        throw(ArgumentError("Unsupported default_sem_weights=:$default_sem_weights"))
    # assemble a list of weighted losses and check params equality
    terms = Vector{LossTerm}()
    params = !isnothing(params) ? copy(params) : params
    has_sem_weights = false
    nsems = 0
    for inp_term in loss_terms
        if inp_term isa AbstractLoss # term
            term = inp_term
            term_w = nothing
            term_id = nothing
        elseif inp_term isa Pair
            if inp_term[1] isa AbstractLoss # term => weight
                term, term_w = inp_term
                term_id = nothing
            elseif inp_term[2] isa AbstractLoss # id => term
                term_id, term = inp_term
                term_w = nothing
            elseif inp_term[2] isa Pair # id => term => weight
                term_id, (term, term_w) = inp_term
                isa(term, AbstractLoss) || throw(
                    ArgumentError(
                        "AbstractLoss expected as a second argument of a loss term double pair (id => loss => weight), $(nameof(typeof(term))) found",
                    ),
                )
            end
        elseif inp_term isa LossTerm
            term_id = id(inp_term)
            term = loss(inp_term)
            term_w = weight(inp_term)
        else
            "[id =>] AbstractLoss [=> weight] expected as a loss term, $(nameof(typeof(inp_term))) found" |>
            ArgumentError |>
            throw
        end

        if term isa SemLoss
            nsems += 1
            has_sem_weights |= !isnothing(term_w)
            # check integrity
            if isnothing(params)
                params = SEM.params(term)
            elseif params != SEM.params(term)
                # FIXME the suggestion might no longer be relevant, since ParTable also stores params order
                """
                The parameters of your SEM models do not match.
                Maybe you tried to specify models of an ensemble via ParameterTables.
                In that case, you may use RAMMatrices instead.
                """ |> error
            end
            check_observed_vars(term)
        elseif !(term isa AbstractLoss)
            "AbstractLoss term expected at $(length(terms)+1) position, $(nameof(typeof(term))) found" |>
            ArgumentError |>
            throw
        end
        push!(terms, LossTerm(term, term_id, term_w))
    end
    isnothing(params) && throw(ErrorException("No SEM models provided."))

    if !has_sem_weights && nsems > 1
        # set the weights of SEMs in the ensemble
        if default_sem_weights == :nsamples
            # weight SEM by the number of samples
            nsamples_total = sum(nsamples(term) for term in terms if issemloss(term))
            for (i, term) in enumerate(terms)
                if issemloss(term)
                    terms[i] =
                        LossTerm(loss(term), id(term), nsamples(term) / nsamples_total)
                end
            end
        elseif default_sem_weights == :uniform # uniform weights
            for (i, term) in enumerate(terms)
                if issemloss(term)
                    terms[i] = LossTerm(loss(term), id(term), 1 / nsems)
                end
            end
        elseif default_sem_weights == :one # do nothing
        end
    end

    terms_tuple = Tuple(terms)
    return Sem{typeof(terms_tuple)}(terms_tuple, params)
end

function Sem(;
    specification = ParameterTable,
    observed::O = SemObservedData,
    implied::I = RAM,
    loss::L = SemML,
    kwargs...,
) where {O, I, L}
    kwdict = Dict{Symbol, Any}(kwargs...)

    set_field_type_kwargs!(kwdict, observed, implied, loss, O, I)

    loss = get_fields!(kwdict, specification, observed, implied, loss)

    return Sem(loss...)
end

############################################################################################
# functions
############################################################################################

params(model::AbstractSem) = model.params
param_labels(model::AbstractSem) = params(model)  # alias

"""
    loss_terms(model::AbstractSem)

Returns a tuple of all [`LossTerm`](@ref) weighted terms in the SEM model.

See also [`sem_terms`](@ref), [`loss_term`](@ref).
"""
loss_terms(model::AbstractSem) = model.loss_terms
nloss_terms(model::AbstractSem) = length(loss_terms(model))

"""
    sem_terms(model::AbstractSem)

Returns a tuple of all weighted SEM terms in the SEM model.

In comparison to [`loss_terms`](@ref) that returns all model terms, including e.g.
regularization terms, this function returns only the [`SemLoss`] terms.

See also [`loss_terms`](@ref), [`sem_term`](@ref).
"""
sem_terms(model::AbstractSem) = Tuple(term for term in loss_terms(model) if issemloss(term))
nsem_terms(model::AbstractSem) = sum(issemloss, loss_terms(model))

nsamples(model::AbstractSem) =
    sum(term -> issemloss(term) ? nsamples(term) : 0, loss_terms(model))

"""
    loss_term(model::AbstractSem, id::Any) -> AbstractLoss

Returns the loss term with the specified `id` from the `model`.
Throws an error if the model has no term with the specified `id`.

See also [`loss_terms`](@ref).
"""
function loss_term(model::AbstractSem, id::Any)
    for term in loss_terms(model)
        if SEM.id(term) == id
            return loss(term)
        end
    end
    error("No loss term with id=$id found")
end

"""
    sem_term(model::AbstractSem, [id]) -> SemLoss

Returns the SEM loss term with the specified `id` from the `model`.
Throws an error if the model has no term with the specified `id` or
if it is not of a [`SemLoss`](@ref) type.

If no `id` is specified and the model contains only one SEM term, the term is returned.
Throws an error if the model contains multiple SEM terms.

See also [`loss_term`](@ref), [`sem_terms`](@ref).
"""
function sem_term(model::AbstractSem, id::Any)
    term = loss_term(model, id)
    issemloss(term) || error("Loss term with id=$id ($(typeof(term))) is not a SEM term")
    return term
end

function sem_term(model::AbstractSem, _::Nothing = nothing)
    if nsem_terms(model) != 1
        error(
            "Model contains $(nsem_terms(model)) SEM terms, you have to specify a specific term",
        )
    end
    for term in loss_terms(model)
        issemloss(term) && return loss(term)
    end
    error("Unreachable reached")
end

# check that all models use the same single loss function
# returns the type of the single SEM loss function, SemLoss if there are multiple different SEM losses,
# nothing if there are no SEM terms.
# If throw_error=true, throws an error if there are multiple different SEM loss functions
check_same_semterm_type(model::AbstractSem; throw_error::Bool = true) =
    check_same_semterm_type(sem_terms(model); throw_error = throw_error)

# check that all models use the same single loss function
# returns the type of the single SEM loss function,
# nothing if there are multiple different SEM losses or no SEM terms.
# If throw_error=true, throws an error if there are multiple different SEM loss functions
function check_same_semterm_type(terms::Tuple; throw_error::Bool = true)
    isempty(terms) && return nothing

    _semloss(term::SemLoss) = _unwrap(term)
    _semloss(term::LossTerm) = _semloss(loss(term))
    _semloss(term) = throw(ArgumentError("SemLoss term expected, $(typeof(term)) found"))
    _semloss_label(i::Integer, _::Union{SemLoss, LossTerm{<:SemLoss, Nothing}}) = "#$i"
    _semloss_label(i::Integer, term::LossTerm{<:SemLoss, Symbol}) = "#$i ($(SEM.id(term)))"

    term1 = _semloss(terms[1])
    L = typeof(term1).name

    # check that all SemLoss terms are of the same class (ML, FIML, WLS etc), ignore typeparams
    for (i, term) in enumerate(terms)
        lossterm = _semloss(term)
        @assert lossterm isa SemLoss
        if typeof(lossterm).name != L
            if throw_error
                error(
                    "SemLoss term $(_semloss_label(i, term)) is $(typeof(lossterm).name), expected $L. Heterogeneous loss functions are not supported",
                )
            else
                return nothing
            end
        end
    end

    # return the type of the first SEM term
    # note that type params of the SEM terms might be different
    return typeof(term1)
end

# wrappers arounds a single SemLoss term
observed(model::AbstractSem, id::Nothing = nothing) = observed(sem_term(model, id))
implied(model::AbstractSem, id::Nothing = nothing) = implied(sem_term(model, id))
vars(model::AbstractSem, id::Nothing = nothing) = vars(implied(model, id))
observed_vars(model::AbstractSem, id::Nothing = nothing) = observed_vars(implied(model, id))
latent_vars(model::AbstractSem, id::Nothing = nothing) = latent_vars(implied(model, id))

function set_field_type_kwargs!(kwargs, observed, implied, loss, O, I)
    kwargs[:observed_type] = O <: Type ? observed : typeof(observed)
    kwargs[:implied_type] = I <: Type ? implied : typeof(implied)
    if loss isa SemLoss
        kwargs[:loss_types] =
            [aloss isa SemLoss ? typeof(aloss) : aloss for aloss in loss.functions]
    elseif applicable(iterate, loss)
        kwargs[:loss_types] = [aloss isa SemLoss ? typeof(aloss) : aloss for aloss in loss]
    else
        kwargs[:loss_types] = [loss isa SemLoss ? typeof(loss) : loss]
    end
end

# construct Sem fields
function get_fields!(kwargs, spec, observed, implied, loss)
    if !isa(spec, SemSpecification)
        spec = spec(; kwargs...)
    end

    # observed
    if !isa(observed, SemObserved)
        observed = if spec isa EnsembleParameterTable
            Dict(
                term_id => observed(; specification = term_spec, kwargs...) for
                (term_id, term_spec) in pairs(spec.tables)
            )
        else
            observed(; specification = spec, kwargs...)
        end
    end

    # implied
    if !isa(implied, SemImplied)
        # FIXME remove this implicit logic
        # SemWLS only accepts vech-ed implied covariance
        if isa(loss, Type) && (loss <: SemWLS) && !haskey(kwargs, :vech)
            implied_kwargs = copy(kwargs)
            implied_kwargs[:vech] = true
        else
            implied_kwargs = kwargs
        end
        implied = if spec isa EnsembleParameterTable
            Dict(
                term_id => implied(term_spec; implied_kwargs...) for
                (term_id, term_spec) in pairs(spec.tables)
            )
        else
            implied(spec; implied_kwargs...)
        end
    end

    # loss
    loss_kwargs = copy(kwargs)
    loss_kwargs[:nparams] = nparams(spec)
    loss = build_SemTerms(loss, observed, implied; loss_kwargs...)

    return loss
end

# construct loss field
function build_SemTerms(loss, observed, implied; kwargs...)
    function build_SemLoss(aloss, observed, implied)
        if loss isa AbstractLoss
            return loss
        elseif aloss <: SemLoss{O, I} where {O, I}
            return aloss(observed, implied; kwargs...)
        else
            return aloss(; kwargs...)
        end
    end

    if loss isa SemLoss
        return loss
    elseif applicable(iterate, loss)
        return [build_SemLoss(aloss, observed, implied) for aloss in loss]
    else
        if isa(observed, AbstractDict) && isa(implied, AbstractDict)
            observed_ids = Set(keys(observed))
            implied_ids = Set(keys(implied))
            if observed_ids != implied_ids
                """"
                The term ids of the observed and the implied data do not match.
                Observed term ids: $(observed_ids), implied term ids: $(implied_ids)
                """ |>
                ArgumentError |>
                throw
            end
            loss_out = [
                begin
                    term_implied = implied[term_id]
                    if observed_vars(term_observed) != observed_vars(term_implied)
                        "observed_vars differ between the observed and the implied for the term $term_id" |>
                        ArgumentError |>
                        throw
                    end
                    LossTerm(
                        build_SemLoss(loss, term_observed, term_implied),
                        term_id,
                        nothing,
                    )
                end for (term_id, term_observed) in pairs(observed)
            ]
            return loss_out
        else
            if observed_vars(observed) != observed_vars(implied)
                "observed_vars differ between the observed and the implied" |>
                ArgumentError |>
                throw
            end
            return (build_SemLoss(loss, observed, implied),)
        end
    end
end

function update_observed(sem::Sem, new_observed; kwargs...)
    new_terms = Tuple(
        update_observed(lossterm.loss, new_observed; kwargs...) for
        lossterm in loss_terms(sem)
    )
    return Sem(new_terms...)
end

##############################################################
# pretty printing
##############################################################

function Base.show(io::IO, sem::AbstractSem)
    println(io, "Structural Equation Model ($(nameof(typeof(sem))))")
    println(io, "- $(nparams(sem)) parameters")
    println(io, "- Loss terms:")
    for term in loss_terms(sem)
        print(io, "  - ")
        print(io, term)
        println(io)
    end
end
