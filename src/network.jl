export GraphStruct, Eton00, Eton01, Eton10, Eton11

## Spherical harmonics setup ##
"""
Functions to compute spherical harmonics and cartesian bases
Ys(l,rv,r) computes harmonics for rank-l and directions rv
cart_basis() uses Clebsch-Gordon coefficients to compute a basis
"""
function Ys(l,rv,r)
    x = rv[:,1]; y = rv[:,2]; z = rv[:,3]
    if l==0
        harm = .5*sqrt(1/pi) .*x.^0
        return harm
    elseif l==1
        harm = hcat(sqrt(3/(4*pi)).*y./r,
                    sqrt(3/(4*pi)).*z./r,
                    sqrt(3/(4*pi)).*x./r)
        return replace!(harm,NaN=>0)
    elseif l==2
        harm = hcat(.5*sqrt(15/pi).*x.*y./r.^2,
                    .5*sqrt(15/pi).*z.*y./r.^2,
                    .25*sqrt(5/pi).*(-x.^2-y.^2+2*z.^2)./r.^2,
                    .5*sqrt(15/pi).*x.*z./r.^2,
                    .25*sqrt(15/pi).*(x.^2 .-y.^2)./r.^2)
        return replace!(harm,NaN=>0)
    else
        error("Do not have rank ",l," harmonics stored")
    end
end 

## TODO: this works but I don't really know why... ##
function cart_basis()
    inds = [(3,1,1), (2,2,1), (1,3,1),
            (2,1,2), (1,2,2),
            (3,1,3), (1,3,3),
            (3,2,4), (2,3,4),
            (1,1,5), 
            (2,1,6), (1,2,6),
            (3,1,7), (2,2,7), (1,3,7),
            (3,2,8), (2,3,8),
            (3,3,9)]
    coef = [√3,-√3,√3,
            √2,-√2,
            √2,-√2,
            √2,-√2,
            1,
            √2,√2,
            √6,√(3/2),√6,
            √2,√2,
            1].^-1
    CG = zeros(3,3,9); CG[CartesianIndex.(inds)].=coef
    T = [ 1/√2 -im/√2 0;
          0     0     1;
         -1/√2 -im/√2 0]
    c2r = Array(Complex.(I(9).*1.))
    c2r[2,[2,4]].=im/√2; c2r[4,[2,4]].=1/√2 .*[1,-1]
    c2r[5,[5,9]].=im/√2 .*[1,-1]; c2r[9,[5,9]].=1/√2
    c2r[6,[6,8]].=im/√2; c2r[8,[6,8]].=1/√2 .*[1,-1]
    cartbasis = real(cat(((c2r*[transpose(T)*CG[:,:,i]*T for i=1:9]).*
                [1;-im*ones(3);ones(5)])...,dims=3))
end

## Graph storage ##
"""
Stores quantites associated with the graph
i,j are the non-zero indices of the adjacency matrix (not stored explicitly)
    Stored as vectors of length NNZ
p are the locations of the points (N x 3)
rv stores the distance vector between the ijth points (NNZ x 3)
r is the magnitude of rv (NNZ x 1)
N = number of nodes in the graph
Y contains the spherical harmonics of each rv pair, up to a finite rank
cartbasis is the cartesian rank-2 tensor basis for the spherical harmonics.
    T_ij = Y_lm * cartbasis_lmij is the cartesian tensor from the Y_lm coefficients
sp is a sparse matrix (N x NNZ)
    The kth column of sp has one non-zero entry at the ith row given by i[k]
    sp*α maps a list of pairwise features α to nodal features by summing over edges
    The entries in sp can be boolean for a simple sum
        or weighted by the graph (normalized) Laplacian

    Given a vector of edge weights β of size (NNZ x 1) and boolean sp, 
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
    sp::AbstractArray
    r::AbstractArray{T}
    Y::AbstractArray
    cartbasis::AbstractArray{T}
    N::Int
end

function GraphStruct(i::AbstractVector,j::AbstractVector,p::AbstractArray)
    N = size(p,1)
    NNZ = length(i)

    deg = dropdims(sum(sparse(i,j,1),dims=2),dims=2) .-1
    L = -ones(NNZ); L[i.==j] .= deg[i[i.==j]] # graph laplacian
    Ln = -1 ./sqrt.(deg[i].*deg[j]); Ln[i.==j] .= 1 # normalized laplacian
    sp = sparse(tr(i.==(1:N)')).*Float32.(Ln)'

    rv = p[i,:].-p[j,:]
    r = sqrt.(sum(abs2,rv,dims=2))

    Y = [Ys(i,rv,r) for i=0:2]
    cartbasis = permutedims(cart_basis(),[3,1,2])

    GraphStruct(i,j,p,sp,r,Y,cartbasis,N)
end

Flux.@functor GraphStruct
Flux.trainable(m::GraphStruct) = ()

#####################
## Layers ##
"""
Layers are defined using their input and ouput tensor ranks
e.g. Eton00 accepts and ouputs scalar features
     Eton01 accepts scalars and outputs vectors

All layers have a trainable weight array (c_in x c_out x p)
where the p dimension is used as a polynomial expansion basis 
for radial variations in the filter

There is one weight array for each tensor rank used in the layer
"""
#####################
"""
Layer operations:
    1. Compute c_in x c_out for each r_ij by contracting the p dimension
        Note the r = 0 intercept can be non-zero as this is a scalar filter
    2. Apply the filters to the input X via contraction along c_in
    3. Convolve with sp
    4. Apply NL and bias
"""
struct Eton00{T,F}
    weight::AbstractArray{T}
    bias::AbstractVector{T} 
    σ::F
    ch::Pair{<:Integer,<:Integer}
end

function Eton00(ch::Pair{<:Integer,<:Integer}, p, σ = identity;
    init=Flux.glorot_uniform, T::DataType=Float32, bias::Bool=false)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    w = T.(init(ch[1], ch[2], p+1))
    Eton00(w, b, σ, ch)
end

Flux.@functor Eton00

function (e::Eton00)(Xgs::Tuple{AbstractArray{T},GraphStruct{T}}) where {T}
    X, gs = Xgs

    Rp = gs.r.^(0:size(e.weight,3)-1)'
    w = e.weight
    # f = Flux.batched_mul(Rp,permutedims(w,[3,1,2]))
    @ein f[nnz,ci,co] := Rp[nnz,p]*w[ci,co,p]

    Xj = X[gs.j,:] #cat(X[gs.j,:],X[gs.i,:],dims=2)
    # Z = Flux.batched_mul(permutedims(f,[3,2,1]),adim(tr(Xj),2))[:,1,:]|>tr
    @ein Z[nnz,co] := f[nnz,ci,co]*Xj[nnz,ci]
   
    out = gs.sp*Z
    out = e.σ.(out.+tr(e.bias))
    
    return (out,gs)
end

#####################
"""
Layer operations:
    1. Compute c_in x c_out for each r_ij by contracting the p dimension
        Note the r = 0 intercept must be zero
    2. Multiply with spherical harmonics to "point" the filter in the rv direction
    3. Apply the filters to the input X via contraction along c_in
    4. Convolve with sp
    5. Apply NL and bias to magnitude of output and multiply
"""
struct Eton01{T,F}
    weight::AbstractArray{T}
    bias::AbstractVector{T}
    σ::F
    ch::Pair{<:Integer,<:Integer}
end

function Eton01(ch::Pair{<:Integer,<:Integer}, p, σ = identity;
    init=Flux.glorot_uniform, T::DataType=Float32, bias::Bool=false)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    w = T.(init(ch[1], ch[2], p))
    Eton01(w, b, σ, ch)
end

Flux.@functor Eton01

function (e::Eton01)(Xgs::Tuple{AbstractArray{T},GraphStruct{T}}) where {T}
    X, gs = Xgs

    Rp = gs.r.^(1:size(e.weight,3))'
    w = e.weight
    @ein f[nnz,ci,co] := Rp[nnz,p]*w[ci,co,p]

    rh = gs.Y[2][:,[3,1,2]]
    f = f.*adim(adim(rh,2),2)

    Xj = X[gs.j,:]
    @ein Z[nnz,co,d] := f[nnz,ci,co,d]*Xj[nnz,ci]
    
    sp = gs.sp
    out = reshape(sp*reshape(Z,size(Z,1),size(Z,2)*size(Z,3)),size(sp,1),size(Z,2),size(Z,3))
    # @ein out[n,co,d] := sp[n,nnz]*Z[nnz,co,d]
    out = e.σ.(mag(out).+tr(e.bias)).*out
    
    return (out,gs)
end

#####################
"""
Layer operations:
    1. Compute c_in x c_out for each r_ij by contracting the p dimension
        Note the r = 0 intercept must be zero
    2. Multiply with spherical harmonics to "point" the filter in the rv direction
    3. Apply the filters to the input X via contraction along c_in and d for inner product
    4. Convolve with sp
    5. Apply NL and bias to magnitude of output and multiply
"""
struct Eton10{T,F}
    weight::AbstractArray{T}
    bias::AbstractVector{T}
    σ::F
    ch::Pair{<:Integer,<:Integer}
end

function Eton10(ch::Pair{<:Integer,<:Integer}, p, σ = identity;
    init=Flux.glorot_uniform, T::DataType=Float32, bias::Bool=false)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    w = T.(init(ch[1], ch[2], p))
    Eton10(w, b, σ, ch)
end

Flux.@functor Eton10

function (e::Eton10)(Xgs::Tuple{AbstractArray{T},GraphStruct{T}}) where {T}
    X, gs = Xgs

    Rp = gs.r.^(1:size(e.weight,3))'
    w = e.weight
    @ein f[nnz,ci,co] := Rp[nnz,p]*w[ci,co,p]

    rh = gs.Y[2][:,[3,1,2]]
    f = f.*adim(adim(rh,2),2)

    Xj = X[gs.j,:,:]
    @ein Z[nnz,co] := f[nnz,ci,co,d]*Xj[nnz,ci,d]
    
    out = gs.sp*Z
    out = e.σ.(out.+tr(e.bias))
    
    return (out,gs)
end

#####################
"""
Layer operations:
    1. Compute c_in x c_out for each r_ij and each rank by contracting the p dimension
        Note the r = 0 intercept must be zero for rank-1,2
    2. Multiply with spherical harmonics to "point" the filter in the rv direction
    3. Apply the filters to the input X via contraction along c_in 
        and d2 to apply the cartesian tensor operator
    5. Convolve with sp
    6. Apply NL and bias to magnitude of output and multiply
"""
struct Eton11{T,F}
    weight0::AbstractArray{T}
    weight1::AbstractArray{T}
    weight2::AbstractArray{T}
    bias::AbstractVector{T}
    σ::F
    ch::Pair{<:Integer,<:Integer}
end

function Eton11(ch::Pair{<:Integer,<:Integer}, p, σ = identity;
    init=Flux.glorot_uniform, T::DataType=Float32, bias::Bool=false)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    w0 = T.(init(ch[1], ch[2], p+1))
    w1 = T.(init(ch[1], ch[2], p))
    w2 = T.(init(ch[1], ch[2], p))
    Eton11(w0, w1, w2, b, σ, ch)
end

Flux.@functor Eton11

function (e::Eton11)(Xgs::Tuple{AbstractArray{T},GraphStruct{T}}) where {T}
    X, gs = Xgs

    w0,w1,w2 = e.weight0, e.weight1, e.weight2
    Rp = gs.r.^(0:size(e.weight0,3)-1)'
    @ein f0[nnz,ci,co] := Rp[nnz,p]*w0[ci,co,p]
    Rp = Rp[:,2:end]
    @ein f1[nnz,ci,co] := Rp[nnz,p]*w1[ci,co,p]
    @ein f2[nnz,ci,co] := Rp[nnz,p]*w2[ci,co,p]

    f0 = f0.*adim(adim(adim(gs.Y[1],2),2),2)
    f1 = f1.*adim(adim(gs.Y[2],2),2)
    f2 = f2.*adim(adim(gs.Y[3],2),2)
    f = cat(f0,f1,f2,dims=4)
    f = sum(f.*adim(adim(adim(gs.cartbasis,1),1),1),dims=4)[:,:,:,1,:,:]

    Xj = X[gs.j,:,:]
    @ein Z[nnz,co,d1] := f[nnz,ci,co,d1,d2]*Xj[nnz,ci,d2]
    
    sp = gs.sp
    out = reshape(sp*reshape(Z,size(Z,1),size(Z,2)*size(Z,3)),size(sp,1),size(Z,2),size(Z,3))
    # @ein out[n,co,d] := sp[n,nnz]*Z[nnz,co,d]
    out = e.σ.(mag(out).+tr(e.bias)).*out
    
    return (out,gs)
end

#####################