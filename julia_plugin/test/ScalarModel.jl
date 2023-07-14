

println("PHASM: Julia: Loading ScalarModel.jl")

function infer(inputs)
    for input in inputs
        println("From Julia callee: Input: $(input)")
    end

    update = [22.0]
    output = [33.0]
    println("From Julia callee: Output: $(update)")
    println("From Julia callee: Output: $(output)")

    return ([update, output], true)

end # function infer
