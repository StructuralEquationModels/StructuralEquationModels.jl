# Release Process

This file documents the release process of StructuralEquationModels.jl.

## Versioning

We use semantic versioning for our releases. Every release has a three digit number, e.g. `2.15.7`. We follow recommendations described [here](https://julialang.org/blog/2019/08/release-process/) and [here](https://pkgdocs.julialang.org/v1/compatibility/).

### Patch releases

- increment the last digit of the version number
- contain only bug fixes, low-risk performance improvements, and documentation updates
- performance issues can also considered to be bugs
- orthogonal features (features that are *very* unlikely to change any existing code, e.g. not using popular function names) are allowed

### Minor releases

- increment the middle digit of the version number
- new features, minor changes, refactoring of internals
- minor changes: are unlikely to break someones code and dont break the tests

### Major releases

- can break anything
- change API

### Pre 1.0.0

Any version 0.x.x is considered beta.
Note that in semantic versioning pre `1.0.0` releases have different compatibility.
`0.6.x` is not compatible to `0.5.x` but Julia deviates here from semantic versioning and considers e.g. `0.5.2` to be compatible to `0.5.3`.
See [Pkg.jl documentation](https://pkgdocs.julialang.org/v1/compatibility/#compat-pre-1.0) for more infos.

## Workflow
We follow the Gitflow workflow described [here](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) and [here](https://nvie.com/posts/a-successful-git-branching-model/) in large parts. We have the branches

### main
- contains the recent released version
- all commits are tagged

### devel
- contains all changes until a new minor/major release branche is created
- contains the complete history of the project

### release/xxx
- forked from devel to start a new release cycle
- no new features (only bug fixes, documentation generation, etc.)
- is merged into main and tagged to create a new release
- also merged into devel when release is created

### feature/xxx: 
- forked from devel to develop a new feature
- never directly interacts with main; merged into devel if the feature is complete

### hotfix/xxx
- forked from main
- merged into main and devel (or the current release branch) and main is tagged

### documentation/xxx
- remember to add an apropriate tag if merged into main/release (e.g. v1.2.3+doc1 see [Documenter.jl docs](http://mortenpi.eu/Documenter.jl/dev/man/hosting/#Documentation-Versions)
- forked from main/release/devel merged into main/release/devel.
- **only** changes to documentation

### testing/xxx
- changes that do not touch the code but only test
- **only** changes that fix faulty tests or increase test coverage
- forked from main/release/devel merged into main/release


