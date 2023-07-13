
include("Phasm.jl")

module TestModule

println("PHASM: Julia: Loading TestModel.jl")
using Main.Phasm

function infer(model::Model)
    println("Inside TestModule::infer")
    for i in 1:phasm_modelvars_count(model)
        ptr = phasm_modelvars_inputdata(model,i-1)
        println("Model var before: $(unsafe_string(phasm_modelvars_getname(model,i-1))): $(ptr): $(unsafe_load(ptr))")
        unsafe_store!(ptr, 33.0)
        println("Model var after: $(unsafe_string(phasm_modelvars_getname(model,i-1))): $(ptr): $(unsafe_load(ptr))")
    end
end # function infer
end # module TestModule
