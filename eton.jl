using NearestNeighbors
using Distributions
using Plots
using Flux
using GeometricFlux
using SparseArrays
using LinearAlgebra
using Statistics
using Zygote
using IterTools
using CUDA
using Random
using BSON: @save, @load

include("helpers.jl")
include("network.jl")

###################################################################################################
## Generate Graph ##

xbounds = [0,pi]
ybounds = [0,pi]

# N = 100
# bp = [[rand(xbounds,N)';rand(Uniform(ybounds...),N)']';
#       [rand(Uniform(xbounds...),N)';rand(ybounds,N)']'] 

# M = 800
# dp = [rand(Uniform(xbounds...),M)';rand(Uniform(ybounds...),M)']'

# loc = [bp;dp]

M = 32
xx,yy = meshgrid(range(0,stop=pi,length=M))
locx = [xx[:]';yy[:]']'

p = Array(locx)
bound = (p[:,1].==xbounds[1]).|(p[:,1].==xbounds[2]).|(p[:,2].==ybounds[1]).|(p[:,2].==xbounds[2])

k = 16
r = pi/(M/2)

i, j, tree, idxs = build_graph(p,r,false)

gs = GraphStruct(i,j,p)

###################################################################################################
## Generate Data ##

# true solution from RHS coeffs
function datagen(A,p)
    m1 = reshape(0:size(A,1)-1,:,1)
    m2 = reshape(0:size(A,2)-1,1,:)
    B = A./((m1.+1).^2 .+ (m2.+1).^2)
    f = fgen(A,p); fmax = maximum(abs.(f)); f = f./fmax
    u = fgen(B,p)./fmax
    return f, u
end

# Test point
@load "fcs.bson" fcs
# fcs = Agen(16,16)
f, ut = datagen(fcs,p)

data = [datagen(Agen(16,16),p) for i=1:100]

###################################################################################################
## Model ##

encoder = Dense(2,16)
eton = Chain(Eton00(gs,16=>16,16,relu,sigmoid),
             Eton01(gs,16=>16,16,relu,sigmoid),
             Eton11(gs,16=>16,16,relu,sigmoid),
             Eton10(gs,16=>16,16,relu,sigmoid),
             Eton00(gs,16=>16,16,relu))
net = eton
decoder = Dense(16,1)

function model(f)
    X = hcat(f,bound)
    X = encoder(X|>tr)|>tr
    Y = net(X)
    out = decoder(Y|>tr)|>tr
end

function loss(f,ut)
    up = model(f)
    l = mean(abs2,ut.-up)
end               
             
###################################################################################################
## Training ##

opt = ADAM(5e-4)
# @load "weights.bson" weights 
# Flux.loadparams!([encoder,net,decoder],weights)

data = gpu(data)
bound = gpu(bound)
encoder = gpu(encoder)
eton = gpu(eton)
net = gpu(net)
decoder = gpu(decoder)
f, ut = gpu(f), gpu(ut)
CUDA.allowscalar(false)

ps = Flux.params(encoder,net,decoder)
test_data = [(f,ut)]
train_data = ncycle(data,10)
function cb() 
    @show(loss(f,ut))
    # save_data(p,f,ut)
end
function save_data(p,f,ut)
    p,f,ut = cpu(p), cpu(f), cpu(ut)
    X, Y, Z1, Z2, Z3, p = p[:,1],p[:,2],f,ut,cpu(model(gpu(f))),p
    @save "data.bson" X Y Z1 Z2 Z3 p
end

@time cb()
@time Flux.train!(loss, ps, train_data, opt, cb=cb)
save_data(p,f,ut)
weights = cpu.(Flux.params([encoder,net,decoder]))
@save "weights.bson" weights opt