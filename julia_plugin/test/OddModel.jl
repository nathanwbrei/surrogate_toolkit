
println("PHASM: Julia: Loading OddModel.jl")

function infer(inputs)
    for input in inputs
        println("From Julia callee: Input: $(input)")
    end

    entries_squared = inputs[1].^2
    sum_of_squares = [sum(entries_squared)]

    println("From Julia callee: Output: $(entries_squared)")
    println("From Julia callee: Output: $(sum_of_squares)")

    return ([entries_squared, sum_of_squares], true)

end # function infer
