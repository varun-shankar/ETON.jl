export adim, build_graph, tr, mag, R, addGraph, rmGraph

function adim(x,i=0)
    if i==0; i=ndims(x)+1; end
    x = reshape(x,(size(x)...,1))
    dims = [1:i-1;ndims(x);i:ndims(x)-1]
    x = permutedims(x,dims)
end

function build_graph(p,rk;knn=false)
    tree = BallTree(p')
    if !knn
        idxs = inrange(tree, p', rk)
    else
        idxs,_ = NearestNeighbors.knn(tree, p', rk)
    end

    i = vcat([fill(i,length(idxs[i])) for i=1:length(idxs)]...)
    j = vcat(idxs...)

    return i, j, tree, idxs
end

function tr(X)
    if ndims(X)==1
        adim(X,1)
    else
        permutedims(X,[2,1])
    end
end

mag(X) = sqrt.(sum(X.^2,dims=3)[:,:,1])

R(θ,φ,γ)=[1 0       0     ;
          0 cos(θ) -sin(θ);
          0 sin(θ)  cos(θ)]*[ cos(φ) 0 sin(φ);
                              0      1 0     ;
                             -sin(φ) 0 cos(φ)]*[cos(γ) -sin(γ) 0;
                                                sin(γ)  cos(γ) 0;
                                                0       0      1]

addGraph(gs) = x->(x,gs)
rmGraph(xgs) = xgs[1]