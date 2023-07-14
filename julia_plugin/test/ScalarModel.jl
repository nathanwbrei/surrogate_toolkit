

println("PHASM: Julia: Loading TestModel.jl")

function infer(inputs)
    for input in inputs
        println("Input: $(input)")
    end

    update = [22.0]
    output = [33.0]
    println("Output: $(update)")
    println("Output: $(output)")

    return ([update, output], true)

end # function infer
