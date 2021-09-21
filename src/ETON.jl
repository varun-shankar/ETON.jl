module ETON

using NearestNeighbors
using Distributions
using SparseArrays, LinearAlgebra
using OMEinsum; allow_loops(false)
using Statistics
using Flux, Zygote

include("utils.jl")
include("network.jl")

end # module
