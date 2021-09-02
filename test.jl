using NearestNeighbors
using Distributions
using Plots
using Flux
using SparseArrays
using LinearAlgebra
using OMEinsum; allow_loops(false)
using Statistics
using Zygote
using IterTools
using CUDA
using Random
using SEM
using Setfield
using UnPack
using BSON: @save, @load
using NPZ

include("helpers.jl")
include("network.jl")

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
r = .25-eps(Float32)

i, j, tree, idxs = build_graph(p,r)

gs = GraphStruct(i,j,p)

###################################################################################################
## Generate Data ##

function trueU(a,b)
    varForcing(f,x,y,t) = @. sin(2*pi*x*a)+cos(2*pi*y*b)
    s = @set sch.setForcing = varForcing
    d = Diffusion(bc,m1,s,Tf=0.0,dt=0.00)
    sim!(d)
    utrue = d.fld.u
end

data = [(.3*rand(),.3*rand()) for i=1:500]

###################################################################################################
## Model ##

# encoder = Dense(1,64)
eton = Chain(Eton01(gs,1=>8,3,swish),
             Eton11(gs,8=>8,3,swish),
             Eton11(gs,8=>8,3,swish), mag
             Eton00(gs,8=>8,3,swish),
             Eton00(gs,8=>8,3,swish),
             Eton00(gs,8=>8,3,swish),
             Eton00(gs,8=>8,3,swish),
             Eton00(gs,8=>8,3,swish),
             Eton00(gs,8=>1,3))
# decoder = Dense(64,1)
# net = Chain(x->tr(x),encoder,x->tr(x),
#             eton,
#             x->tr(x),decoder,x->tr(x))
net = SkipConnection(eton,+)

# eton2 = Chain(Eton00(gs,1=>16,3,swish),
#               Eton00(gs,16=>16,3,swish),
#               Eton00(gs,16=>16,3,swish),
#               Eton00(gs,16=>1,3))
# net2 = Chain(net,SkipConnection(eton2,+))

function opLearn!(dfn::Diffusion)
    @unpack rhs,ν,msh,fld = dfn
    @unpack u,ub = fld

    rhs = l2g(rhs./ν,msh)
    shape = size(rhs)
    u = reshape(net(reshape(rhs,:,1)),shape...)
    u = g2l(u,msh)
    u = SEM.mask(u,fld.M)

    u = u + ub
    @pack! dfn.fld = u
    return
end

function model(a,b)
    varForcing(f,x,y,t) = @. sin(2*pi*x*a)+cos(2*pi*y*b)
    sch_so = @set sch.solve! = opLearn!
    sch_so = @set sch_so.setForcing = varForcing
    dfn_so = Diffusion(bc,m1,sch_so,Tf=0.0,dt=0.00)
    sim!(dfn_so)
    upred = dfn_so.fld.u
end

function loss(data...)
    upred = model(data...)
    utrue = trueU(data...)
    mean(abs2,upred.-utrue)
end            
             
###################################################################################################
## Training ##

function cb()
    @show loss(test_data...)
    # plt = meshplt(trueU(test_data...),m1); plt = meshplt!(m1.x,m1.y,model(test_data...),c=:blue);
    # display(plt)
    # IJulia.clear_output(true);
end

ps = Flux.params(eton)
test_data = (0.,0.)
opt = ADAM(1e-3)
Flux.train!(loss,ps,data,opt, cb = Flux.throttle(cb,1))
plt = meshplt(trueU(test_data...),m1); plt = meshplt!(m1.x,m1.y,model(test_data...),c=:blue); display(plt)
@show loss(test_data...)