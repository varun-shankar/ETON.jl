using Revise
using ETON
using Flux
using Statistics, StatsBase
using Zygote
using IterTools
using CUDA
using Random
using BSON: @save, @load
using NPZ

###################################################################################################
## Generate Graph ##
frac = 0.1
case = "BUMP_h20"

x = npzread("/home/opc/data/kaggle_data/kepsilon/kepsilon_"*case*"_Cx.npy")
y = npzread("/home/opc/data/kaggle_data/kepsilon/kepsilon_"*case*"_Cy.npy")
z = npzread("/home/opc/data/kaggle_data/kepsilon/kepsilon_"*case*"_Cz.npy")

N = size(x,1)
# inds = LinearIndices((1:Int(sqrt(N)), 1:Int(sqrt(N))))[1:2:end,1:2:end][:]
inds = sample(1:N,Int(floor(frac*N)),replace=false,ordered=true)
# inds = Bool.(sum(y.==(sort(unique(y))[1:2:end])',dims=2)[:].*
#              sum(z.==(sort(unique(z))[1:2:end])',dims=2)[:])|>findall
# inds = 1:N
x = x[inds]
y = y[inds]
z = z[inds]
p = hcat(x,y,z)
# p = (R(pi/4,0,0)*p')'

k = 4

i, j, tree, idxs = build_graph(p,k,knn=true)

gs = GraphStruct(i,j,p)
gs = gpu(gs)

###################################################################################################
## Generate Data ##

input = npzread("/home/opc/data/kaggle_data/kepsilon/kepsilon_"*case*"_I.npy")[inds,:]
tensors = npzread("/home/opc/data/kaggle_data/kepsilon/kepsilon_"*case*"_Tensors.npy")[inds,:,:,:]
l = npzread("/home/opc/data/kaggle_data/labels/"*case*"_k.npy")[inds]
labels = l./maximum(l)
baseline = npzread("/home/opc/data/kaggle_data/kepsilon/kepsilon_"*case*"_k.npy")[inds]./maximum(l)

function data_load(cases)
    [(npzread("/home/opc/data/kaggle_data/kepsilon/kepsilon_"*case*"_I.npy")[inds,:],
      begin 
        l = npzread("/home/opc/data/kaggle_data/labels/"*case*"_k.npy")[inds]
        l./maximum(l)
      end) for case in cases]
end
###################################################################################################
## Model ##

h = 8
encoder = Chain(Flux.normalise, Dense(47,16,swish),Dense(16,h,tanh))
eton = Chain(Eton00(gs,h=>h,8,swish,bias=true),
             Eton01(gs,h=>h,8,swish,bias=true),
             Eton11(gs,h=>h,8,swish,bias=true),
             mag,#Eton10(gs,h=>h,8,swish,bias=true),
             Eton00(gs,h=>h,8,tanh,bias=true))
decoder = Chain(Dense(h,16,swish),Dense(16,1))
net = Chain(x->tr(x),encoder,x->tr(x),
            SkipConnection(eton,+),
            x->tr(x),decoder,x->tr(x))

function model(X)
    # X = adim(X,2)
    Y = net(X)
    # Y = sum(Y.*tensors,dims=2)[:,1,:,:]
end

function loss(X,Y)
    Yp = model(X)
    l = mean(abs2,Y.-Yp)
end               
             
###################################################################################################
## Training ##

opt = ADAM(1e-3)
# @load "w.bson" weights
# Flux.loadparams!([net],weights)

input = gpu(input)
net = gpu(net)
input, tensors, labels = gpu(input), gpu(tensors), gpu(labels)
CUDA.allowscalar(false)

ps = Flux.params(net)
test_data = (input,labels)
cases = [case]
        # ["DUCT_1150","DUCT_1250"]#,"DUCT_1300","DUCT_1350","DUCT_1400","DUCT_1500","DUCT_1600"]#,
        #  "DUCT_1800","DUCT_2000","DUCT_2205","DUCT_2400"]
train_data = ncycle(data_load(cases)|>gpu,100)
function cb() 
    @show(loss(test_data...))
    # save_data(p,f,ut)
end
function save_data(p,input,labels,baseline)
    p,input,labels,baseline = cpu(p), cpu(input), cpu(labels), cpu(baseline)
    X, Y, Z, I, Lh, L, B = p[:,1],p[:,2],p[:,3],input,cpu(model(gpu(input))),labels, baseline
    @save "data.bson" X Y Z I Lh L B
end

cb()
@time Flux.train!(loss, ps, train_data, opt, cb=Flux.throttle(cb,2))
save_data(p,input,labels,baseline)
weights = cpu.(Flux.params([net]))
@save "w.bson" weights opt

# grad = gradient(()->loss(test_data...),ps)
# for p in ps
#     display(grad[p])
# end