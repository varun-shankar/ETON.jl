using ETON
using Flux
using Statistics
using Zygote

###################################################################################################
## Generate Graph ##

N = 100
p = rand(N,3)

r = .25
i, j, tree, idxs = build_graph(p,r,knn=false)
gs = GraphStruct(i,j,p) |> gpu

###################################################################################################
## Generate Data ##

Y = p[:,1] + p[:,2] + p[:,3] |> gpu     # Arbitrary function

X = adim(p,2) |> gpu    # Position vector input (N points x 1 feature x 3 dims)

###################################################################################################
## Model ##

net = Chain(addGraph(gs),
            Eton11(1=>32,4,swish),
            Eton10(32=>32,4,swish),
            Eton00(32=>32,4,swish),
            Eton00(32=>32,4,swish),
            Eton00(32=>1,4),
            rmGraph) |> gpu

function model(X)
    Y = net(X)
end

function loss(X,Y)
    Yp = model(X)
    l = mean(abs2,Y.-Yp)
end

###################################################################################################
## Training ##

opt = ADAM(1e-3)
ps = Flux.params(net)

println("Initial loss    : ",loss(X,Y))
for i=1:100; Flux.train!(loss, ps, [(X,Y)], opt); end
println("Final loss      : ",loss(X,Y))

## Test equivariance
p = (R((rand(3).*2*pi)...)*p')'
i, j, tree, idxs = build_graph(p,r,knn=false)
gs = GraphStruct(i,j,p) |> gpu
X = adim(p,2) |> gpu
println("Invariance check: ",loss(X,Y))