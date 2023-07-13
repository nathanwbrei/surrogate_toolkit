
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
    @ccall phasm_modelvars_getinputdata(model::Model,index::Int64)::Ptr{Float64}
end

function phasm_modelvars_setoutputdata(model::Model, index, ptr, len) 
    println("In phasm_modelvars_setoutputdata")
    @ccall phasm_modelvars_setoutputdata(model::Model,index::Int64,ptr::Ptr{Float64},Csize_t(len)::Csize_t)::Cvoid
    println("Finished phasm_modelvars_setoutputdata")
end






end # module Phasm
