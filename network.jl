## Graph storage ##
struct GraphStruct{T}
    i::AbstractArray
    j::AbstractArray
    p::AbstractArray{T}
    rv::AbstractArray{T}
    r::AbstractArray{T}
    θ::AbstractArray{T}
    N::Int
    NN::AbstractArray
    sp::AbstractArray
end

function GraphStruct(i::AbstractVector,j::AbstractVector,p::AbstractArray)
    N = size(p,1)
    NN = dropdims(sum(sparse(i,j,1),dims=2),dims=2)
    rv = p[i,:].-p[j,:]
    r = sqrt.(sum(abs2,rv,dims=2))
    θ = atan.(rv[:,2],rv[:,1])
    sp = sparse(Float32.(tr(i.==(1:N)')))
    GraphStruct(i,j,p,rv,r,θ,N,NN,sp)
end

Flux.@functor GraphStruct
Flux.trainable(m::GraphStruct) = ()

## Layers ##
#####################

struct Eton00{T,Po,R,F,G<:GraphStruct}
    weight::AbstractArray{T}
    bias::AbstractVector{T}
    pointwise::Po   
    radial::R    
    σ::F
    gs::G
    ch::Pair{<:Integer,<:Integer}
end

function Eton00(gs::GraphStruct, ch::Pair{<:Integer,<:Integer}, pointwise, radial, σ = identity;
    init=Flux.glorot_uniform, T::DataType=Float32, bias::Bool=false)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    Eton00(T.(init(ch[1], ch[2])),b, pointwise, radial, σ, gs, ch)
end

function Eton00(gs::GraphStruct, ch::Pair{<:Integer,<:Integer}, hl::Int = 16, 
    γ = identity, σ = identity; kwargs...)
    pointwise = Chain(Dense(ch[1],hl,γ), Dense(hl,ch[2]))
    radial = Chain(Dense(1,hl,γ), Dense(hl,ch[2]))
    Eton00(gs, ch, pointwise, radial, σ; kwargs...)
end

Flux.@functor Eton00

function (e::Eton00)(X::AbstractArray{T}) where {T}
    # Linear transform input features
    Xo = X*e.weight
    # Apply pointwise weighting
    Y1 = Xo.*tr(e.pointwise(X|>tr))
    # Calculate RBF
    G = tr(e.radial(e.gs.r|>tr))
    # Apply radial function and convolve from graph
    Y2 = e.gs.sp*(Xo[e.gs.j,:].*G)
    # Non-linearity
    out = e.σ.((Y1.+Y2))
    return out
end
# sp*X[j]
# i = [1,1,2,3]
# j = [2,3,2,3]
# r = [1,2,3,4]
# sp = [1,1,0,0;
#       0,0,1,0;
#       0,0,0,1]
# X = [10,20,30]
# [0 1 1;
#  0 1 0;
#  0 0 1]

# Aw =   [0 1 2;
#         0 3 0;
#         0 0 4]
# Aw*X == sp*(r.*X[j]) = sp*([20
#                             60
#                             60
#                             120])
# [80
#  60
#  120]
#####################

struct Eton01{T,Fi,R,F,G<:GraphStruct}
    weight::AbstractArray{T}
    bias::AbstractVector{T}
    filter::Fi # 2 x fo
    radial::R  
    σ::F
    gs::G
    ch::Pair{<:Integer,<:Integer}
end

function Eton01(gs::GraphStruct, ch::Pair{<:Integer,<:Integer}, radial, σ = identity;
    init=Flux.glorot_uniform, T::DataType=Float32, bias::Bool=false)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    Eton01(T.(init(ch[1], ch[2])), b, T.(init(2, ch[2])), radial, σ, gs, ch)
end

function Eton01(gs::GraphStruct, ch::Pair{<:Integer,<:Integer}, hl::Int = 16, 
    γ = identity, σ = identity; kwargs...)
    radial = Chain(Dense(1,hl,γ), Dense(hl,ch[2]))
    Eton01(gs, ch, radial, σ; kwargs...)
end

Flux.@functor Eton01

function (e::Eton01)(X::AbstractArray{T}) where {T}
    # Linear transform input features
    Xo = X*e.weight # N x fo
    # Rotate each filter
    filter = e.filter./(sqrt.(sum(e.filter.^2,dims=1)))
    F = Flux.stack(batchmul.([rot(e.gs.θ)],Flux.unstack(adim(filter,2),3)),4) # 2 x 1 x N x fo
    F = dropdims(permutedims(F,[3,4,1,2]),dims=4) # N x fo x 2
    # Calculate RBF
    R = tr(e.radial(e.gs.r|>tr)) # N x fo
    # Radially weight filters
    G = F.*R # N x fo x 2
    # Apply filters via scalar multiplication
    Y = Xo[e.gs.j,:].*G # N x fo x 2
    # Convolve
    Y = Flux.stack([e.gs.sp].*Flux.unstack(Y,3),3)
    # Non-linearity
    out = e.σ.(sqrt.(sum(Y.^2,dims=3))).*Y
    return out
end

#####################

struct Eton10{T,Fi,R,F,G<:GraphStruct}
    weight::AbstractArray{T}
    bias::AbstractVector{T}
    filter::Fi
    radial::R
    σ::F
    gs::G
    ch::Pair{<:Integer,<:Integer}
end

function Eton10(gs::GraphStruct, ch::Pair{<:Integer,<:Integer}, radial, σ = identity;
    init=Flux.glorot_uniform, T::DataType=Float32, bias::Bool=false)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    Eton10(T.(init(ch[1], ch[2])), b, T.(init(2, ch[2])), radial, σ, gs, ch)
end

function Eton10(gs::GraphStruct, ch::Pair{<:Integer,<:Integer}, hl::Int = 16, 
    γ = identity, σ = identity; kwargs...)
    radial = Chain(Dense(1,hl,γ), Dense(hl,ch[2]))
    Eton10(gs, ch, radial, σ; kwargs...)
end

Flux.@functor Eton10

function (e::Eton10)(X::AbstractArray{T}) where {T}
    # Linear transform input features
    Xo = batchmul(X,e.weight)
    # Rotate each filter
    filter = e.filter./(sqrt.(sum(e.filter.^2,dims=1)))
    F = Flux.stack(batchmul.([rot(e.gs.θ)],Flux.unstack(adim(filter,2),3)),4) # 2 x 1 x N x fo
    F = dropdims(permutedims(F,[3,4,1,2]),dims=4) # N x fo x 2
    # Calculate RBF
    R = tr(e.radial(e.gs.r|>tr)) # N x fo
    # Radially weight filters
    G = F.*R # N x fo x 2
    # Apply filters via inner product
    Y = dropdims(sum(Xo[e.gs.j,:,:].*G,dims=3),dims=3) # N x fo
    # Convolve
    Y = e.gs.sp*Y
    # Non-linearity
    out = e.σ.(Y)
    return out
end

#####################

struct Eton11{T,Fi,R,F,G<:GraphStruct}
    weight::AbstractArray{T}
    bias::AbstractVector{T}
    filter::Fi
    radial::R
    σ::F
    gs::G
    ch::Pair{<:Integer,<:Integer}
end

function Eton11(gs::GraphStruct, ch::Pair{<:Integer,<:Integer}, radial, σ = identity;
    init=Flux.glorot_uniform, T::DataType=Float32, bias::Bool=false)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    Eton11(T.(init(ch[1], ch[2])), b, T.(init(2,2, ch[2])), radial, σ, gs, ch)
end

function Eton11(gs::GraphStruct, ch::Pair{<:Integer,<:Integer}, hl::Int = 16, 
    γ = identity, σ = identity; kwargs...)
    radial = Chain(Dense(1,hl,γ), Dense(hl,ch[2]))
    Eton11(gs, ch, radial, σ; kwargs...)
end

Flux.@functor Eton11

function (e::Eton11)(X::AbstractArray{T}) where {T}
    # Linear transform input features
    Xo = batchmul(X,e.weight)
    # Rotate each filter
    filter = e.filter./(adim(adim(vcat([sum(diag(e.filter[:,:,i]),dims=1) for i=1:e.ch[2]]...),1),1))
    F = Flux.stack(batchmul.([rot(e.gs.θ)],Flux.unstack(filter,3)),4) # 2 x 2 x N x fo
    F = permutedims(F,[3,4,1,2]) # N x fo x 2 x 2
    # Calculate RBF
    R = tr(e.radial(e.gs.r|>tr)) # N x fo
    # Radially weight filters
    G = F.*R # N x fo x 2 x 2
    # Apply filters via tensor product
    Y = batchmul(permutedims(G,[3,4,1,2]),permutedims(adim(Xo[e.gs.j,:,:]),[3,4,1,2])) # 2 x 1 x N x fo
    Y = dropdims(permutedims(Y,[3,4,1,2]),dims=4) # N x fo x 2
    # Convolve
    Y = Flux.stack([e.gs.sp].*Flux.unstack(Y,3),3)
    # Non-linearity
    out = e.σ.(sqrt.(sum(Y.^2,dims=3))).*Y
    return out
end

#####################

function sparseCu(i::AbstractVector,j::AbstractVector,v::CuVector{T},m::Int,n::Int) where T
    sp = sparse(i|>cpu,j|>cpu,1:length(v),m,n)
    sp = CUDA.CUSPARSE.CuSparseMatrixCSC(cu(sp.colptr),cu(sp.rowval),v[sp.nzval],(m,n))
    # CUDA.CUSPARSE.CuSparseMatrixCSR(sp)
end
Zygote.@adjoint function sparseCu(i,j,v,m,n)
    sparseCu(i,j,v,m,n), Δ -> (nothing,nothing,Δ[CartesianIndex.(i,j)],nothing,nothing)
end