"""
    start_val(model)
    
Return a vector of starting values.
Defaults are FABIN 3 starting values for single models and simple starting values for
ensemble models.
"""
function start_val end
# Single Models ----------------------------------------------------------------------------

# splice model and loss functions
start_val(model::AbstractSemSingle; kwargs...) = start_val(
    model,
    model.observed,
    model.imply,
    model.optimizer,
    model.loss.functions...;
    kwargs...,
)

# Fabin 3 starting values for RAM(Symbolic)
start_val(model, observed, imply, optimizer, args...; kwargs...) =
    start_fabin3(model; kwargs...)

# Ensemble Models --------------------------------------------------------------------------
start_val(model::SemEnsemble; kwargs...) = start_simple(model; kwargs...)
