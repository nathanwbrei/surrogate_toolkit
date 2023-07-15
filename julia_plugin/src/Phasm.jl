
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
    data_ptr::Vector{Ptr{Float64}} = [0]
    shape_ptr::Vector{Ptr{Int64}} = [0]
    ndims::Vector{Csize_t} = [0]
    @ccall phasm_modelvars_getinputdata(
        model::Model,
        index::Int64,
        pointer(data_ptr)::Ptr{Ptr{Float64}},
        pointer(shape_ptr)::Ptr{Ptr{Int64}},
        pointer(ndims)::Ptr{Csize_t}
        )::Cvoid
    shape_vec = unsafe_wrap(Array, shape_ptr[1], (ndims[1],); own=false)
    shape_tup = (shape_vec...,)
    return unsafe_wrap(Array, data_ptr[1], shape_tup; own=false)
end

function phasm_modelvars_setoutputdata(model::Model, index, array) 
    ptr = pointer(array)
    dims = Csize_t(ndims(array))
    if (dims == 1)
        len = Csize_t(length(array))
        @ccall phasm_modelvars_setoutputdata(model::Model,index::Int64,ptr::Ptr{Float64},len::Csize_t)::Cvoid
    else
        shape = [Int64(x) for x in size(array)]::Vector{Int64}
        @ccall phasm_modelvars_setoutputdata2(model::Model,index::Int64,ptr::Ptr{Float64},pointer(shape)::Ptr{Int64}, dims::Csize_t)::Cvoid
    end
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
