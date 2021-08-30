function adim(x,i=0)
    if i==0; i=ndims(x)+1; end
    x = reshape(x,(size(x)...,1))
    dims = [1:i-1;ndims(x);i:ndims(x)-1]
    x = permutedims(x,dims)
end

function batchmul(A, B)
    Adims = size(A); Bdims = size(B)
    if ndims(A) > 3
        A = reshape(A,(Adims[1:2]...,:))
    end
    if ndims(B) > 3
        B = reshape(B,(Bdims[1:2]...,:))
    end
    enddims = ndims(A)>=ndims(B) ? Adims[3:end] : Bdims[3:end]
    out = reshape(Flux.batched_mul(A,B),(Adims[1],Bdims[2],enddims...))
end

function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where T
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repeat(vx, m, 1), repeat(vy, 1, n))
end
meshgrid(v::AbstractVector) = meshgrid(v, v)

# generate fourier coeffs
function Agen(m,n)
    A = 2 .*rand(m,n) .-1
    m1 = reshape(0:size(A,1)-1,:,1)
    m2 = reshape(0:size(A,2)-1,1,:)
    A = A.*exp.(-1 .*(m1.+m2)).*rand(size(A)...) # decay with wavenumber + noise
end

# field from fourier coeffs
function fgen(A,p)
    m1 = reshape(0:size(A,1)-1,:,1)
    m2 = reshape(0:size(A,2)-1,1,:)
    x = reshape(p[:,1],1,1,:)
    y = reshape(p[:,2],1,1,:)

    f = @. A*sin((m1+1)*x)*sin((m2+1)*y)
    f = reshape(sum(f,dims=(1,2)),size(p,1))
end

function build_graph(p,rk,keep_diags=true,k=false)
    tree = BallTree(p')
    if !k
        idxs = inrange(tree, p', rk)
    else
        idxs,_ = knn(tree, p', rk)
    end

    i = vcat([fill(i,length(idxs[i])) for i=1:length(idxs)]...)
    j = vcat(idxs...)

    if !keep_diags
        ods = findall(i.!==j)
        i = i[ods]
        j = j[ods]
    end
    return i, j, tree, idxs
end

function tr(X)
    if ndims(X)==1
        adim(X,1)
    else
        permutedims(X,[2,1])
    end
end

function rot(θ)
    θ = adim(adim(θ,1),1)
    [cos.(θ) -sin.(θ);
     sin.(θ) cos.(θ)]
end

mag(X) = sqrt.(sum(X.^2,dims=3)[:,:,1])