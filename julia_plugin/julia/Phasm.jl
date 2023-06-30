
"""
# module Phasm
- Julia version: 1.9
- Author: nbrei
- Date: 2023-05-31
"""

module Phasm
export Model, phasm_modelvars_count, phasm_modelvars_getname, phasm_modelvars_isinput, phasm_modelvars_isoutput, phasm_modelvars_inputdata, phasm_modelvars_outputdata
println("PHASM: Julia: Loading Phasm.jl")

struct OpaqueModel
end
const Model = Ptr{OpaqueModel}

phasm_modelvars_count(model::Model) = @ccall phasm_modelvars_count(model::Model)::Int64
phasm_modelvars_getname(model::Model, index) = @ccall phasm_modelvars_getname(model::Model,index::Int64)::Cstring
phasm_modelvars_isinput(model::Model, index) = @ccall phasm_modelvars_isinput(model::Model,index::Int64)::Bool
phasm_modelvars_isoutput(model::Model, index) = @ccall phasm_modelvars_isoutput(model::Model,index::Int64)::Bool
phasm_modelvars_inputdata(model::Model, index) = @ccall phasm_modelvars_inputdata(model::Model,index::Int64)::Ptr{Float64}
phasm_modelvars_outputdata(model::Model, index) = @ccall phasm_modelvars_outputdata(model::Model,index::Int64)::Ptr{Float64}

end # module Phasm
