mutable struct PartiallyTyped{T, C}
        typed::T
        calculatable::C
        nottyped
end

function PartiallyTyped(x)
        # constructor for needed arguments
        # first argument could be e.g. ram
        # second argument is calculated from first (e.g. obs_cov)
        # third is left empty
        PartiallyTyped(x, x^2, nothing)
end

# create object via constructor above
test = PartiallyTyped(3.0)

# show that functions are type stable
function stable_func(x::PartiallyTyped)
        x.calculatable + 1
end

stable_func(test)
@code_warntype stable_func(test)

# nottyped fields can be set to any value
function stable_func!(x::PartiallyTyped)
        setfield!(x, :nottyped, stable_func(x))
end

stable_func!(test)
test.nottyped

@code_warntype stable_func!(test)

# nottyped may be reused later
# but code relying on nottyped fields will be slow (but there are remedies)

unstable_func(x::PartiallyTyped)= x.nottyped * 4
@code_warntype unstable_func(test)

# alternative constructors are possible
function PartiallyTyped(; calculatable)
        PartiallyTyped(nothing, calculatable, nothing)
end

# e.g. only supply obs_cov
test2 = PartiallyTyped(calculatable = 5)
# still typestabe, with full inference
stable_func(test2)
@code_warntype stable_func(test2)
stable_func!(test2)
test2.nottyped
