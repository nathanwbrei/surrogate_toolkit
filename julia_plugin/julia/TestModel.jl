
include("Phasm.jl")

module TestModule

println("PHASM: Julia: Loading TestModel.jl")
using Main.Phasm

function infer(model::Model)
    println("Inside TestModule::infer")
    ptr = phasm_modelvars_getinputdata(model,0)
    println("Reading model variable '$(phasm_modelvars_getname(model,0))' at index 0: $(ptr): $(unsafe_load(ptr))")

    result = [22.0]
    println("Writing model variable '$(phasm_modelvars_getname(model,0))' at index 0")
    phasm_modelvars_setoutputdata(model,0, pointer(result), length(result))

    result = [33.0]
    println("Writing model variable '$(phasm_modelvars_getname(model,1))' at index 1")
    phasm_modelvars_setoutputdata(model,1, pointer(result), length(result))

    return true

end # function infer
end # module TestModule
