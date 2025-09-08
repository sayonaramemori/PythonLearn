import deepxde as dde
import numpy as np
import torch
from pde_define import pde_rans

torch.set_default_dtype(torch.float64)
dde.config.set_default_float("float64")
dde.config.set_random_seed(666)
dde.model.optimizers.config.set_LBFGS_options(maxiter=990000);

Length = 1.0
Vel = 1.0
Re = 100.0
# Define the domain
domain = dde.geometry.Rectangle([0, 0], [Length, Length])  # 2D domain (0, 1) x (0, 1)

# timedomain = dde.geometry.TimeDomain(0, 30)  # Time domain [0, 1]
# domain = dde.geometry.GeometryXTime(domain,timedomain)
# Boundary conditions
# Input is x,y,t
def boundary_up(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1],Length)
def boundary_not_up(x, on_boundary):
    return on_boundary and not dde.utils.isclose(x[1],Length)

device = torch.device('cuda')
# Is this right?
def initial_condition(x):
    res = torch.DoubleTensor(1024,1)
    print(x.shape)
    for i in range(1024):
        if dde.utils.isclose(x[i][1],Length):
            res[i][0] = Vel
        else:
            res[i][0] = 0.0
    return res.to(device);

def pressure_bc_hori(x,on_boundary):
    return on_boundary and (dde.utils.isclose(x[1],0) or dde.utils.isclose(x[1],Length))
def pressure_bc_vert(x,on_boundary):
    return on_boundary and (dde.utils.isclose(x[0],0) or dde.utils.isclose(x[0],Length))
def robin_bc_vert(x,y,_):
    res = dde.grad.jacobian(y,x,i=2,j=0)
    return res
def robin_bc_hori(x,y,_):
    res = dde.grad.jacobian(y,x,i=2,j=1)
    return res

bc_up_u = dde.icbc.DirichletBC(domain,lambda x: Vel, boundary_up,component=0)
bc_other_u = dde.icbc.DirichletBC(domain,lambda x: 0.0, boundary_not_up,component=0)
bc_v= dde.icbc.DirichletBC(domain,lambda x: 0.0, lambda _,on_boundary: on_boundary,component=1)
bc_p_vert = dde.icbc.boundary_conditions.OperatorBC(domain,robin_bc_vert, pressure_bc_vert)
bc_p_hori = dde.icbc.boundary_conditions.OperatorBC(domain,robin_bc_hori, pressure_bc_hori)

ic_u= dde.icbc.IC(domain, initial_condition,lambda x, on_initial: on_initial ,component=0)
ic_v= dde.icbc.IC(domain, lambda x: 0,lambda x, on_initial: on_initial ,component=1)
# ic3= dde.icbc.IC(domain, lambda x: 0,lambda x, on_initial: on_initial ,component=2)

layer_size = [2] + [64] * 6 +  [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.pytorch.FNN(layer_size, activation, initializer)

# Create the model
data = dde.data.PDE(
# data = dde.data.TimePDE(
    domain,
    pde_rans,
    [bc_up_u,bc_other_u,bc_v,bc_p_vert,bc_p_hori],
    num_domain=2560,
    num_boundary=256,
    num_test=2560,
    # num_initial=1024
)
model = dde.Model(data, net)

# Compile the model
# model.compile("adam", lr=1e-3)
# model.compile("L-BFGS", lr=1e-3, metrics=["l2 relative error"])
model.compile("L-BFGS")
# model.restore("./model/top_cover_model.ckpt-59000.pt")

# Train the model
checkpointer = dde.callbacks.ModelCheckpoint( "./model/top_cover_model.ckpt", verbose=1, save_better_only=True)
pde_resampler = dde.callbacks.PDEPointResampler(period=100)
losshistory, train_state = model.train(callbacks=[checkpointer,pde_resampler])
from mailmsg import send_over_msg
send_over_msg()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
