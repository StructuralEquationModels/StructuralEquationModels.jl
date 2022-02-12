using CSV, DataFrames

date = string(ARGS...)

a = rand(10, 10)

a = DataFrame(a, :auto)

CSV.write("test_"*date*".csv", a, delim = ";")