# RANS-PINN demo for a 2D top-lid-driven cavity using DeepXDE (PyTorch backend)
# ----------------------------------------------------------------------------
# - Geometry: unit square [0, 1] x [0, 1]
# - Boundary: no-slip on all walls; top lid moves with U=1 along +x
# - Model: steady incompressible RANS with standard k-ω (Wilcox) two-equation model
# - Unknowns: u(x,y), v(x,y), p(x,y), k(x,y), ω(x,y)
# - Eddy viscosity: ν_t = k / ω
# - Production: P_k = ν_t * (2*u_x^2 + 2*v_y^2 + (u_y + v_x)^2)
# - Diffusion uses variable-coefficient Laplacian: ∇·[(ν + σ * ν_t) ∇(·)]
# NOTE: This is a **didactic demo** intended to show how to set up a RANS-PINN in DeepXDE.
#       It is not tuned/validated for high-Re accuracy. You may need to tweak
#       loss weights, training samples, and boundary formulations for robust results.
# ----------------------------------------------------------------------------

import numpy as np
import deepxde as dde
from deepxde.backend import torch

# Use PyTorch backend (must come before other DeepXDE ops if set dynamically)
dde.backend.set_default_backend("pytorch")

# -------------------------
# Physical & model settings
# -------------------------
U_lid = 1.0          # top lid speed
L = 1.0              # cavity length scale
Re = 10000.0         # Reynolds number based on U_lid and L
nu = U_lid * L / Re  # kinematic viscosity

# k-ω (Wilcox 1988) constants
beta_star = 0.09
alpha = 5.0 / 9.0
beta = 3.0 / 40.0
sigma_k = 2.0
sigma_w = 2.0

# Small epsilons for numerical stability
eps = 1e-6

# -------------------------
# Geometry & boundary tags
# -------------------------
geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

# Boundary indicator helpers (x is a 1D numpy array [x,y])
def on_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)


def on_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1.0)


def on_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)


def on_top(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1.0)


# Wall-distance-based omega boundary (very rough, for demo)
# ω_wall ~ C * ν / d^2, where d is the distance to the nearest wall.
# We clip d to avoid singularities exactly at the wall collocation points.
C_omega = 60.0


def omega_wall_func(x):
    # x: (N, 2) numpy array
    xx = x[:, 0:1]
    yy = x[:, 1:2]
    d = np.minimum(np.minimum(xx, 1.0 - xx), np.minimum(yy, 1.0 - yy))
    d = np.maximum(d, 1e-3)  # clip for numerical stability
    return C_omega * nu / (beta_star * d ** 2)


def k_wall_func(x):
    # Low turbulence kinetic energy near wall (Dirichlet)
    return np.full((x.shape[0], 1), 1e-6)


# -------------------------
# PDE residuals (RANS + k-ω)
# -------------------------
# y = [u, v, p, k, w]

def pde(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    k = y[:, 3:4]
    w = y[:, 4:5]

    # Enforce positivity (softplus) for k and w inside PDE to avoid negative ν_t
    k_pos = torch.nn.functional.softplus(k) + eps
    w_pos = torch.nn.functional.softplus(w) + eps

    # First derivatives
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)
    k_x = dde.grad.jacobian(y, x, i=3, j=0)
    k_y = dde.grad.jacobian(y, x, i=3, j=1)
    w_x = dde.grad.jacobian(y, x, i=4, j=0)
    w_y = dde.grad.jacobian(y, x, i=4, j=1)

    # Second derivatives (for diffusion terms)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    k_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)
    k_yy = dde.grad.hessian(y, x, component=3, i=1, j=1)
    w_xx = dde.grad.hessian(y, x, component=4, i=0, j=0)
    w_yy = dde.grad.hessian(y, x, component=4, i=1, j=1)

    # Eddy viscosity and its gradients
    nu_t = k_pos / w_pos
    nu_eff = nu + nu_t

    # Gradients of nu_t (for variable-coefficient diffusion divergence)
    nu_t_x = dde.grad.jacobian(nu_t, x, i=0, j=0)
    nu_t_y = dde.grad.jacobian(nu_t, x, i=0, j=1)

    # Continuity
    cont = u_x + v_y

    # Momentum residuals: u*∂u/∂x + v*∂u/∂y = -∂p/∂x + ∂x[(ν+ν_t)∂u/∂x] + ∂y[(ν+ν_t)∂u/∂y]
    # Implement divergence of variable-coefficient diffusion via product rule
    diff_u = nu_t_x * u_x + (nu + nu_t) * u_xx + nu_t_y * u_y + (nu + nu_t) * u_yy
    diff_v = nu_t_x * v_x + (nu + nu_t) * v_xx + nu_t_y * v_y + (nu + nu_t) * v_yy

    mom_u = u * u_x + v * u_y + p_x - diff_u
    mom_v = u * v_x + v * v_y + p_y - diff_v

    # Turbulence production P_k
    S_term = 2.0 * u_x * u_x + 2.0 * v_y * v_y + (u_y + v_x) * (u_y + v_x)
    P_k = nu_t * S_term

    # k-equation: u·∇k = P_k - β* k ω + ∇·[(ν + σ_k ν_t) ∇k]
    nu_eff_k = nu + sigma_k * nu_t
    nu_eff_k_x = dde.grad.jacobian(nu_eff_k, x, i=0, j=0)
    nu_eff_k_y = dde.grad.jacobian(nu_eff_k, x, i=0, j=1)
    div_diff_k = nu_eff_k_x * k_x + nu_eff_k * k_xx + nu_eff_k_y * k_y + nu_eff_k * k_yy
    conv_k = u * k_x + v * k_y
    res_k = conv_k - P_k + beta_star * k_pos * w_pos - div_diff_k

    # ω-equation: u·∇ω = α (ω/k) P_k - β ω^2 + ∇·[(ν + σ_ω ν_t) ∇ω]
    nu_eff_w = nu + sigma_w * nu_t
    nu_eff_w_x = dde.grad.jacobian(nu_eff_w, x, i=0, j=0)
    nu_eff_w_y = dde.grad.jacobian(nu_eff_w, x, i=0, j=1)
    div_diff_w = nu_eff_w_x * w_x + nu_eff_w * w_xx + nu_eff_w_y * w_y + nu_eff_w * w_yy
    conv_w = u * w_x + v * w_y
    prod_w = alpha * (w_pos / k_pos) * P_k
    res_w = conv_w - prod_w + beta * w_pos * w_pos - div_diff_w

    return [cont, mom_u, mom_v, res_k, res_w]


# -------------------------
# Boundary conditions
# -------------------------
# Velocity: no-slip on all walls, with top lid u=U_lid
bc_u_top = dde.icbc.DirichletBC(geom, lambda x: U_lid * np.ones((x.shape[0], 1)), on_top, component=0)
bc_u_left = dde.icbc.DirichletBC(geom, lambda x: np.zeros((x.shape[0], 1)), on_left, component=0)
bc_u_right = dde.icbc.DirichletBC(geom, lambda x: np.zeros((x.shape[0], 1)), on_right, component=0)
bc_u_bottom = dde.icbc.DirichletBC(geom, lambda x: np.zeros((x.shape[0], 1)), on_bottom, component=0)

bc_v_all = dde.icbc.DirichletBC(geom, lambda x: np.zeros((x.shape[0], 1)), lambda x, on_b: on_b, component=1)

# Turbulence quantities on walls (very rough demo values)
bc_k_wall = dde.icbc.DirichletBC(geom, k_wall_func, lambda x, on_b: on_b, component=3)
bc_w_wall = dde.icbc.DirichletBC(geom, omega_wall_func, lambda x, on_b: on_b, component=4)

# Pressure reference at the cavity center to fix gauge pressure
ref_point = np.array([[0.5, 0.5]])
bc_p_ref = dde.icbc.PointSetBC(ref_point, np.array([[0.0]]), component=2)

bcs = [
    bc_u_top,
    bc_u_left,
    bc_u_right,
    bc_u_bottom,
    bc_v_all,
    bc_k_wall,
    bc_w_wall,
    bc_p_ref,
]

# -------------------------
# Data & model
# -------------------------
# You can increase num_domain / num_boundary for better accuracy (slower training)

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=25000,
    num_boundary=3000,
)

# Network: 2 -> 5 with Tanh activations
net = dde.nn.FNN([2] + [128] * 6 + [5], "tanh", "Glorot normal")

# Optional: scale outputs to help optimization (esp. pressure vs velocities)
# Here we leave default; consider an output transform if training is unstable.

model = dde.Model(data, net)

# Loss weights to balance different PDE residual magnitudes
# [continuity, mom_u, mom_v, res_k, res_w] + BCs (auto-assigned after)
model.compile(
    "adam",
    lr=1e-3,
    loss_weights=[1.0, 1.0, 1.0, 0.3, 0.3],
)

losshistory, train_state = model.train(iterations=20000, display_every=1000)

# Switch to L-BFGS for fine-tuning (often helps PINNs)
model.compile("L-BFGS")
losshistory, train_state = model.train()

# Save the trained model and a sample prediction grid
model.save("rans_pinn_lid_cavity")

# Sample a grid and save fields to .npz for postprocessing
nx, ny = 101, 101
X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
XY = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

pred = model.predict(XY)

u_pred = pred[:, 0].reshape(ny, nx)
v_pred = pred[:, 1].reshape(ny, nx)
p_pred = pred[:, 2].reshape(ny, nx)
k_pred = pred[:, 3].reshape(ny, nx)
w_pred = pred[:, 4].reshape(ny, nx)

np.savez(
    "rans_pinn_lid_cavity_fields.npz",
    x=X,
    y=Y,
    u=u_pred,
    v=v_pred,
    p=p_pred,
    k=k_pred,
    w=w_pred,
    Re=Re,
    nu=nu,
)

print("Saved model checkpoint and fields to: rans_pinn_lid_cavity* files.")

# -------------------------
# Tips for improvement
# -------------------------
# 1) Try curriculum on Re (start smaller, then increase).
# 2) Add interior/near-wall sampling bias (dde.geometry.sampler) to better resolve shear.
# 3) Tune loss weights; consider normalization of each PDE residual.
# 4) Embed boundary conditions with an output transform to reduce BC loss burden.
# 5) Validate against a CFD baseline for quantitative assessment.

