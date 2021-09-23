using Revise
using ETON
using Flux
using DiffEqFlux, DifferentialEquations, DiffEqSensitivity
using Statistics
using Zygote
using IterTools
using CUDA
using Random
using SEM
using Setfield
using BSON: @save, @load
CUDA.allowscalar(false)

###################################################################################################
## Generate Graph ##

Ex = 16; nr1 = 2;
Ey = 16; ns1 = 2;

ifperiodic = [false,false]
m1 = Mesh(nr1,ns1,Ex,Ey,ifperiodic)

bc = ['D','D','D','D']

setIC(u,x,y,t) = 0.0 .*u
setBC(ub,x,y,t) = @. 0+0*x
setForcing(f,x,y,t) = @. 1+0*x
setVisc(ν,x,y,t) = @. 1+0*x

sch = DiffusionScheme(setIC,setBC,setForcing,setVisc)
dfn = Diffusion(bc,m1,sch,Tf=0.0,dt=0.00)

l2g(u,msh) = ABu(msh.Qy',msh.Qx',msh.mult.*u)
g2l(u,msh) = ABu(msh.Qy,msh.Qx,u)

p = hcat(l2g(m1.x,m1)[:],l2g(m1.y,m1)[:])
p = hcat(p,zeros(size(p,1)))
M = l2g(dfn.fld.M,m1)[:]

# k = 16
r = .25#-eps(Float32)

i, j, tree, idxs = build_graph(p,r)

gs = GraphStruct(i,j,p) |> gpu

###################################################################################################
## Generate Data ##

function trueU(a,b)
    varForcing(f,x,y,t) = @. sin(2*pi*x*a)+cos(2*pi*y*b)
    s = @set sch.setForcing = varForcing
    d = Diffusion(bc,m1,s,Tf=0.0,dt=0.00)
    sim!(d)
    utrue = d.fld.u
    f = d.f
    return hcat(l2g(f,m1)[:],M), adim(l2g(utrue,m1)[:])
end

data = [trueU(.3*rand(),.3*rand()) for i=1:50] |> gpu
test_data = trueU(0.,0.) |> gpu

###################################################################################################
## Model ##

h = 16
encoder = Chain(Dense(2,h,swish),Dense(h,h,tanh)) |> gpu
eton = Chain(Eton00(gs,h=>h,8,swish,bias=true),
             Eton00(gs,h=>h,8,swish,bias=true),
             Eton00(gs,h=>h,8,swish,bias=true),
             Eton00(gs,h=>h,8,tanh,bias=true)) |> gpu
de = NeuralODE(eton,(0.f0,1.f0),Tsit5(),
                save_everystep=false,save_start=false,
                reltol=1e-3,abstol=1e-3) |> gpu
decoder = Chain(Dense(h,h,swish),Dense(h,1)) |> gpu
net = Chain(x->tr(x),encoder,x->tr(x),
            de, gpu, x->x[:,:,1],
            x->tr(x),decoder,x->tr(x)) |> gpu


function model(X)
    Y = net(X)
end

function loss(f,utrue)
    upred = model(f)
    mean(abs2,upred.-utrue)
end            
             
###################################################################################################
## Training ##

opt = ADAM(1e-3)
# @load "w.bson" weights
# Flux.loadparams!(net,weights)

function cb()
    @show loss(test_data...)
end
function save_data(p,input,labels,baseline)
    p,input,labels,baseline = cpu(p), cpu(input), cpu(labels), cpu(baseline)
    X, Y, Z, I, Lh, L, B = p[:,1],p[:,2],p[:,3],input,cpu(model(gpu(input))),labels, baseline
    @save "data.bson" X Y Z I Lh L B
end

ps = Flux.params(net)

cb()

train_data = ncycle([test_data],500)
Flux.train!(loss,ps,train_data,opt, cb = Flux.throttle(cb,1))
weights = cpu.(Flux.params(net))
@save "w.bson" weights opt
save_data(p,test_data[1],test_data[2],0)
