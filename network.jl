## Graph storage ##
"""
Stores quantites associated with the graph
i,j are the non-zero indices of the adjacency matrix (not stored explicitly)
    Stored as vectors of length NNZ
p are the locations of the points (N x 2)
rv stores the distance vector between the ijth points (NNZ x 2)
r is the magnitude of rv (NNZ x 1)
θ is the angle between the ijth points (NNZ x 1)
N = number of nodes in the graph
NN is the number of neighbors for each node (N x 1)
sp is a boolean sparse matrix (N x NNZ)
    The kth column of sp has one non-zero entry at the ith row given by i[k]
    sp*α performs a row-wise sum of the sparse mat defined by (i,j,α)

    Given a vector of edge weights β of size (NNZ x 1), 
    sp*β is the equivalent operation to A_β*ones(N), 
    where A_β is the weighted adjecency matrix with weights given by β

    Given a feature X of size (N x 1),
    A_β*X can be computed as sp*(X[j].*β)

    The advantage of this construction is that different weightings
    can be broadcast easily without constructing a new sparse mat:
    [A_β1*X A_β2*X] = sp*(X[j].*[β1 β2])
"""
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

#####################
## Layers ##
"""
Layers are defined using their input and ouput tensor ranks
e.g. Eton00 accepts and ouputs scalar features
     Eton01 accepts scalars and outputs vectors

All layers have a trainable weight matrix (f_in x f_out)
that linearly transforms input features (regardless of rank)

All layers have a trainable RBF that accepts a radius and
outputs a radial weight per output channel
"""
#####################
"""
Applies a pointwise weighting to the output features as well
    Accepts input features at a location and outputs a weight
    for each output feature at the same location
Layer operations:
    1. Linearly transform input features
    2. Generate and apply pointwise weights
    3. Calculate RBF for each connection in the graph
    4. Apply radial weights to the output features
    5. Convolve with sp
    6. Sum pointwise and pairwise and apply NL
"""
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

#####################
"""
Higher order layers need directional filters for each output channel
    For a vector filter, this is stored with an array of size (2 x f_out)
Layer operations:
    1. Linearly transform input features
    2. Normalize vector filters with magnitude
    3. Rotate the filters by θ (angle of rv) for each pairwise connection
    4. Calculate RBF for each connection in the graph
    5. Apply radial weights to filters
    6. Scalar multiply output features by the now weighted vector filters
    5. Convolve with sp
    6. Apply NL to magnitude of the output features
"""
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
    F = Flux.stack(batchmul.([rot(e.gs.θ)],Flux.unstack(adim(filter,2),3)),4) # 2 x 1 x NNZ x fo
    F = dropdims(permutedims(F,[3,4,1,2]),dims=4) # NNZ x fo x 2
    # Calculate RBF
    R = tr(e.radial(e.gs.r|>tr)) # NNZ x fo
    # Radially weight filters
    G = F.*R # NNZ x fo x 2
    # Apply filters via scalar multiplication
    Y = Xo[e.gs.j,:].*G # NNZ x fo x 2
    # Convolve
    Y = Flux.stack([e.gs.sp].*Flux.unstack(Y,3),3) # N x fo x 2
    # Non-linearity
    out = e.σ.(sqrt.(sum(Y.^2,dims=3))).*Y
    return out
end

#####################
"""
Higher order layers need directional filters for each output channel
    For a vector filter, this is stored with an array of size (2 x f_out)
Layer operations:
    1. Linearly transform input features
    2. Normalize vector filters with magnitude
    3. Rotate the filters by θ (angle of rv) for each pairwise connection
    4. Calculate RBF for each connection in the graph
    5. Apply radial weights to filters
    6. Take the inner product of output features and weighted vector filters
    5. Convolve with sp
    6. Apply NL
"""
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
    Xo = batchmul(X,e.weight) # N x fo x 2
    # Rotate each filter
    filter = e.filter./(sqrt.(sum(e.filter.^2,dims=1)))
    F = Flux.stack(batchmul.([rot(e.gs.θ)],Flux.unstack(adim(filter,2),3)),4) # 2 x 1 x NNZ x fo
    F = dropdims(permutedims(F,[3,4,1,2]),dims=4) # NNZ x fo x 2
    # Calculate RBF
    R = tr(e.radial(e.gs.r|>tr)) # NNZ x fo
    # Radially weight filters
    G = F.*R # NNZ x fo x 2
    # Apply filters via inner product
    Y = dropdims(sum(Xo[e.gs.j,:,:].*G,dims=3),dims=3) # NNZ x fo
    # Convolve
    Y = e.gs.sp*Y # N x fo
    # Non-linearity
    out = e.σ.(Y)
    return out
end

#####################
"""
Higher order layers need directional filters for each output channel
    For a tensor filter, this is stored with an array of size (2 x 2 x f_out)
Layer operations:
    1. Linearly transform input features
    2. Normalize tensor filters with trace
    3. Rotate the filters by θ (angle of rv) for each pairwise connection
    4. Calculate RBF for each connection in the graph
    5. Apply radial weights to filters
    6. Multiply weighted tensor filters by vector output features 
    5. Convolve with sp
    6. Apply NL to magnitude of the output features
"""
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
    Xo = batchmul(X,e.weight) # N x fo x 2
    # Rotate each filter
    filter = e.filter./(adim(adim(vcat([sum(diag(e.filter[:,:,i]),dims=1) for i=1:e.ch[2]]...),1),1))
    F = Flux.stack(batchmul.([rot(e.gs.θ)],Flux.unstack(filter,3)),4) # 2 x 2 x NNZ x fo
    F = permutedims(F,[3,4,1,2]) # NNZ x fo x 2 x 2
    # Calculate RBF
    R = tr(e.radial(e.gs.r|>tr)) # NNZ x fo
    # Radially weight filters
    G = F.*R # NNZ x fo x 2 x 2
    # Apply filters via matvec
    Y = batchmul(permutedims(G,[3,4,1,2]),permutedims(adim(Xo[e.gs.j,:,:]),[3,4,1,2])) # 2 x 1 x NNZ x fo
    Y = dropdims(permutedims(Y,[3,4,1,2]),dims=4) # NNZ x fo x 2
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