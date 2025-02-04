# Extending the package

As discussed in the section on [Model Construction](@ref), every Structural Equation Model (`Sem`) consists of three (four with the optimizer) parts:

![SEM concept typed](../assets/concept_typed.svg)

On the following pages, we will explain how you can define your own custom parts and "plug them in". There are certain things you **have to do** to define custom parts and some things you **can do** to have a more pleasent experience. In general, these requirements fall into the categories
- minimal (to use your custom part and fit a `Sem` with it)
- use the outer constructor to build a model in a more convenient way
- use additional functionality like standard errors, fit measures, etc.