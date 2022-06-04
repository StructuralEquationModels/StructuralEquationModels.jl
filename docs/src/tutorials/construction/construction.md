# Model construction

There are two different ways of constructing a SEM in our package. You can use the [Outer Constructor](@ref) oder [Build by parts](@ref).
The final models will be the same, the outer constructor just has some sensible defaults that make your life easier.
All tutorials until now used the outer constructor `Sem(specification = ..., data = ..., ...)`, which is normally the more convenient way.
However, our package is build for extensibility, so there may be cases where **user-defined** parts of a model do not work with the outer constructor.
Therefore, building the model by parts is always available as a fallback.