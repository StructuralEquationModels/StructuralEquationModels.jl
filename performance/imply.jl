## As a function (normal)
struct ram_fun{F <: Function}
    func::F
end

function(ram::ram_fun)(par)
    A = ram.func(par)
end

function three_mean_func(x)
    S =[x[1] 0 0 0 0 0 0 0 0 0 0 0.0
        0 x[2] 0 0 0 0 0 0 0 0 0 0
        0 0 x[3] 0 0 0 0 0 0 0 0 0
        0 0 0 x[4] 0 0 0 0 0 0 0 0
        0 0 0 0 x[5] 0 0 0 0 0 0 0
        0 0 0 0 0 x[6] 0 0 0 0 0 0
        0 0 0 0 0 0 x[7] 0 0 0 0 0
        0 0 0 0 0 0 0 x[8] 0 0 0 0
        0 0 0 0 0 0 0 0 x[9] 0 0 0
        0 0 0 0 0 0 0 0 0    x[10] x[13] x[14]
        0 0 0 0 0 0 0 0 0    x[13] x[11] x[15]
        0 0 0 0 0 0 0 0 0    x[14] x[15] x[12]]

    F =[1.0 0 0 0 0 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 1 0 0 0]

    A =[0  0  0  0  0  0  0  0  0  1     0     0.0
        0  0  0  0  0  0  0  0  0  x[16] 0     0
        0  0  0  0  0  0  0  0  0  x[17] 0     0
        0  0  0  0  0  0  0  0  0  0     1     0
        0  0  0  0  0  0  0  0  0  0     x[18] 0
        0  0  0  0  0  0  0  0  0  0     x[19] 0
        0  0  0  0  0  0  0  0  0  0     0     1
        0  0  0  0  0  0  0  0  0  0     0     x[20]
        0  0  0  0  0  0  0  0  0  0     0     x[21]
        0  0  0  0  0  0  0  0  0  0     0     0
        0  0  0  0  0  0  0  0  0  0     0     0
        0  0  0  0  0  0  0  0  0  0     0     0]

    M = [x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], 0, 0, 0]

    return [S, F, A, M]
end

ram_fun_test = ram_fun(three_mean_func)

## As a function (symbolic)



## full symbolic computation

## parse symbolmatr. to indices
struct ram_ind{T <: AbstractArray, N <: Union{AbstractArray, Nothing},
    B <: BitArray, B2 <: BitArray, V <: Vector{Int64}, IM <: AbstractArray,
    V2 <: Vector{Any}}
    S::T
    F::T
    A::T
    M::N
    Ind_s::B
    Ind_a::B
    Ind_m::B2
    parind_s::V
    parind_a::V
    parind_m::V
    imp_cov::IM
    names::V2
end


function ram_ind(S, F, A, M, start_values, imp_cov)
    Ind_s = eltype.(S) .== Any
    Ind_a = eltype.(A) .== Any

    sfree = sum(Ind_s)
    afree = sum(Ind_a)

    parsym_a = A[Ind_a]
    parsym_s = S[Ind_s]

    parind_s = zeros(Int64, sfree)
    parind_a = zeros(Int64, afree)

    parind_s[1] = 1

    for i = 2:sfree
        parind_s[i] = maximum(parind_s) + 1
        for j = 1:i
            if parsym_s[i] == parsym_s[j]
                parind_s[i] = parind_s[j]
            end
        end
    end

    parind_a[1] = maximum(parind_s) + 1

    for i = 2:afree
        parind_a[i] = maximum(parind_a) + 1
        for j = 1:i
            if parsym_a[i] == parsym_a[j]
                parind_a[i] = parind_a[j]
            end
        end
    end

    if !isa(M, Nothing)
        Ind_m = eltype.(M) .== Any
        mfree = sum(Ind_m)
        parsym_m = M[Ind_m]
        parind_m = zeros(Int64, mfree)


        parind_m[1] = maximum(parind_a) + 1

        for i = 2:mfree
            parind_m[i] = maximum(parind_m) + 1
            for j = 1:i
                if parsym_m[i] == parsym_m[j]
                    parind_m[i] = parind_m[j]
                end
            end
        end

        M_start = copy(M)
        M_start[Ind_m] .= start_values[parind_m]
        M_start = convert(Array{Float64}, M_start)
    else
        M_start = nothing
        Ind_m = bitrand(2,2)
        parind_m = rand(Int64, 5)
    end

    S_start = copy(S)
    A_start = copy(A)


    S_start[Ind_s] .= start_values[parind_s]
    A_start[Ind_a] .= start_values[parind_a]


    return ram_ind(
        convert(Array{Float64}, S_start),
        convert(Array{Float64}, F),
        convert(Array{Float64}, A_start),
        M_start,
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, S_start),
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, F),
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, A_start),
        #convert(Array{ForwardDiff.Dual{Nothing, Float64, 0}}, M_start),
        Ind_s, Ind_a, Ind_m, parind_s, parind_a, parind_m, imp_cov,
        [unique(parsym_s); unique(parsym_a)])
end

function (ram::ram_ind{T, N, B, B2,V, IM,V2} where {
    T <: AbstractArray, N <: AbstractArray,
        B <: BitArray, B2 <: BitArray, V <: Vector{Int64}, IM <: AbstractArray,
        V2 <: Vector{Any}
    })(parameters)
    ram.S[ram.Ind_s] .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
    ram.M[ram.Ind_m] .= parameters[ram.parind_m]
end

function (ram::ram_ind{T, N, B, B2,V, IM,V2} where {
    T <: AbstractArray, N <: AbstractArray,
        B <: BitArray, B2 <: BitArray, V <: Vector{Int64}, IM <: AbstractArray,
        V2 <: Vector{Any}
    })(parameters)
    for i = 1:length(ram.Ind_s)
        ram.S[ram.Ind_s[i]]
     .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
    ram.M[ram.Ind_m] .= parameters[ram.parind_m]
end

function (ram::ram_ind{Array{Float64,2},Nothing})(parameters)
    ram.S[ram.Ind_s] .= parameters[ram.parind_s]
    ram.A[ram.Ind_a] .= parameters[ram.parind_a]
end

S =[:x1 0 0 0 0 0 0 0 0 0 0 0.0
    0 :x2 0 0 0 0 0 0 0 0 0 0
    0 0 :x3 0 0 0 0 0 0 0 0 0
    0 0 0 :x4 0 0 0 0 0 0 0 0
    0 0 0 0 :x5 0 0 0 0 0 0 0
    0 0 0 0 0 :x6 0 0 0 0 0 0
    0 0 0 0 0 0 :x7 0 0 0 0 0
    0 0 0 0 0 0 0 :x8 0 0 0 0
    0 0 0 0 0 0 0 0 :x9 0 0 0
    0 0 0 0 0 0 0 0 0 :x10 :x13 :x14
    0 0 0 0 0 0 0 0 0 :x13 :x11 :x15
    0 0 0 0 0 0 0 0 0 :x14 :x15 :x12]

F =[1.0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0]

A =[0  0  0  0  0  0  0  0  0  1     0     0.0
    0  0  0  0  0  0  0  0  0  :x16 0     0
    0  0  0  0  0  0  0  0  0  :x17 0     0
    0  0  0  0  0  0  0  0  0  0     1     0
    0  0  0  0  0  0  0  0  0  0     :x18 0
    0  0  0  0  0  0  0  0  0  0     :x19 0
    0  0  0  0  0  0  0  0  0  0     0     1
    0  0  0  0  0  0  0  0  0  0     0     :x20
    0  0  0  0  0  0  0  0  0  0     0     :x21
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0
    0  0  0  0  0  0  0  0  0  0     0     0]

M = [:x22, :x23, :x24, :x25, :x26, :x27, :x28, :x29, :x30, 0, 0, 0]

start_val_ind = vcat(fill(1, 10), [0.5, 0.5, 1, 0.5, 1], fill(0.5, 6), fill(1, 10))

ram_ind_test = ram_ind(S, F, A, M, start_val,
    zeros(9,9))

## Test
using BenchmarkTools

@benchmark ram_ind_test(start_val_ind)

@benchmark ram_fun_test(start_val_ind)
