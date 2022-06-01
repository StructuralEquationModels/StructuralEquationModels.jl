# Model construction

There are two different ways of specifying a SEM in our package. You can use the [Outer Constructor](@ref) oder [Build by parts](@ref).
The difference in both contruction methods (building by parts or with the outer constructor) is only about how to arrive at the final model; the choice about which parts to put together is independet from it. However, the outer constructor simply has some default values that are put in place unless you demand something else.
All tutorials until now used the outer constructor `Sem(specification = ..., data = ..., ...)`, which is normally the more convenient way.
However, our package is build for extensibility, so there may be cases where **user-defined** parts of a model do not work with the outer constructor.
Therefore, building the model by parts is always available as a fallback.