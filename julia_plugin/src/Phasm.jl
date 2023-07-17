
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
    untyped_data_ptr::Vector{Ptr{Nothing}} = [0]
    shape_ptr::Vector{Ptr{Int64}} = [0]
    dtype::Vector{Int64} = [0]
    ndims::Vector{Csize_t} = [0]
    @ccall phasm_modelvars_getinputdata(
        model::Model,
        index::Int64,
        pointer(dtype)::Ptr{Int64},
        pointer(untyped_data_ptr)::Ptr{Ptr{Nothing}},
        pointer(shape_ptr)::Ptr{Ptr{Int64}},
        pointer(ndims)::Ptr{Csize_t}
        )::Cvoid
    if dtype[1] == 1
        t = UInt8
    elseif dtype[1] == 2
        t = Int16
    elseif dtype[1] == 3
        t = Int32
    elseif dtype[1] == 4
        t = Int64
    elseif dtype[1] == 5
        t = Float32
    elseif dtype[1] == 6
        t = Float64
    else
        println("Invalid dtype $(dtype)")
    end
    data_ptr = reinterpret(Ptr{t}, untyped_data_ptr)
    shape_vec = unsafe_wrap(Array, shape_ptr[1], (ndims[1],); own=false)
    shape_tup = (shape_vec...,)
    return unsafe_wrap(Array, data_ptr[1], shape_tup; own=false)
end

function phasm_modelvars_setoutputdata(model::Model, index, array) 

    # enum class DType { Undefined, UI8, I16, I32, I64, F32, F64 };
    if eltype(array) == UInt8
        dtype = 1
    elseif eltype(array) == Int16
        dtype = 2
    elseif eltype(array) == Int32
        dtype = 3
    elseif eltype(array) == Int64
        dtype = 4
    elseif eltype(array) == Float32
        dtype = 5
    elseif eltype(array) == Float64
        dtype = 6
    else
        dtype = 0
    end

    ptr = pointer(array)
    dims = Csize_t(ndims(array))
    if (dims == 1)
        len = Csize_t(length(array))
        @ccall phasm_modelvars_setoutputdata(model::Model,index::Int64,dtype::Int32,ptr::Ptr{Float64},len::Csize_t)::Cvoid
    else
        shape = [Int64(x) for x in size(array)]::Vector{Int64}
        @ccall phasm_modelvars_setoutputdata2(model::Model,index::Int64,dtype::Int32,ptr::Ptr{Float64},pointer(shape)::Ptr{Int64}, dims::Csize_t)::Cvoid
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
