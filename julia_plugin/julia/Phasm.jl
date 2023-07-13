
"""
# module Phasm
- Julia version: 1.9
- Author: nbrei
- Date: 2023-05-31
"""

module Phasm
export Model, phasm_modelvars_count, phasm_modelvars_getname, phasm_modelvars_isinput, phasm_modelvars_isoutput, phasm_modelvars_getinputdata, phasm_modelvars_setoutputdata
println("PHASM: Julia: Loading Phasm.jl")

struct OpaqueModel
end
const Model = Ptr{OpaqueModel}

function phasm_modelvars_count(model::Model) 
    return @ccall phasm_modelvars_count(model::Model)::Int64
end

function phasm_modelvars_getname(model::Model, index) 
    return unsafe_string(@ccall phasm_modelvars_getname(model::Model,index::Int64)::Cstring)
end

phasm_modelvars_isinput(model::Model, index) = @ccall phasm_modelvars_isinput(model::Model,index::Int64)::Bool
phasm_modelvars_isoutput(model::Model, index) = @ccall phasm_modelvars_isoutput(model::Model,index::Int64)::Bool

function phasm_modelvars_getinputdata(model::Model, index) 
    ptr = @ccall phasm_modelvars_getinputdata(model::Model,index::Int64)::Ptr{Float64}
    return unsafe_wrap(Array, ptr, (1,); own=false)
end

function phasm_modelvars_setoutputdata(model::Model, index, array) 
    println("In phasm_modelvars_setoutputdata")
    ptr = pointer(array)
    dims = Csize_t(ndims(array))
    if (dims == 1)
        len = Csize_t(length(array))
        @ccall phasm_modelvars_setoutputdata(model::Model,index::Int64,ptr::Ptr{Float64},len::Csize_t)::Cvoid
    else
        shape = [Int64(x) for x in size(array)]::Vector{Int64}
        @ccall phasm_modelvars_setoutputdata2(model::Model,index::Int64,ptr::Ptr{Float64},pointer(shape)::Ptr{Int64}, dims::Csize_t)::Cvoid
    end
    println("Finished phasm_modelvars_setoutputdata")
end

function phasm_infer(model::Model, infer_fn)
    inputs = []
    var_count = phasm_modelvars_count(model)
    for i in 0:var_count-1
        if (phasm_modelvars_isinput(model, i))
            push!(inputs, phasm_modelvars_getinputdata(model, i))
        end
    end

    outputs, is_confident = infer_fn(inputs)
    output_idx = 1  # Julia is 1-indexed!!!
    for i in 0:var_count-1
        if (phasm_modelvars_isoutput(model, i))
            phasm_modelvars_setoutputdata(model, i, outputs[output_idx])
            output_idx += 1
        end
    end
    return is_confident
end # function infer


end # module Phasm
