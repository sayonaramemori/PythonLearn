import torch
import deepxde as dde
import numpy as np

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



def pde_gpt(x, u):
    u_vel = u[:, 0:1]
    v_vel = u[:, 1:2]
    p     = u[:, 2:3]

    # 一阶导
    u_x = dde.grad.jacobian(u_vel, x, i=0, j=0)
    u_y = dde.grad.jacobian(u_vel, x, i=0, j=1)
    v_x = dde.grad.jacobian(v_vel, x, i=0, j=0)
    v_y = dde.grad.jacobian(v_vel, x, i=0, j=1)
    p_x = dde.grad.jacobian(p,     x, i=0, j=0)
    p_y = dde.grad.jacobian(p,     x, i=0, j=1)

    # 二阶导
    u_xx = dde.grad.hessian(u_vel, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(u_vel, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(v_vel, x, component=0, i=0, j=0)
    v_yy = dde.grad.hessian(v_vel, x, component=0, i=1, j=1)

    continuity = u_x + v_y
    momentum_u = u_vel * u_x + v_vel * u_y + p_x - (1/Re) * (u_xx + u_yy)
    momentum_v = u_vel * v_x + v_vel * v_y + p_y - (1/Re) * (v_xx + v_yy)

    return continuity, momentum_u, momentum_v

def pde(x, y):
    u = y[:, 0:1]  # u-velocity component (x-direction)
    v = y[:, 1:2]  # v-velocity component (y-direction)
    
    u_x = dde.grad.jacobian(y, x, i=0,j=0)
    u_y = dde.grad.jacobian(y, x, i=0,j=1)
    v_x = dde.grad.jacobian(y, x, i=1,j=0)
    v_y = dde.grad.jacobian(y, x, i=1,j=1)
    p_x = dde.grad.jacobian(y, x, i=2,j=0)  # Pressure gradient in x
    p_y = dde.grad.jacobian(y, x, i=2,j=1)  # Pressure gradient in y
    
    u_xx = dde.grad.hessian(u, x, i=0, j=0)  # u_xx (second derivative of u with respect to x)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)  # u_yy (second derivative of u with respect to y)
    v_xx = dde.grad.hessian(v, x, i=0, j=0)  # v_xx (second derivative of v with respect to x)
    v_yy = dde.grad.hessian(v, x, i=1, j=1)  # v_yy (second derivative of v with respect to y)
    
    conv_u = u * u_x + v * u_y  # u * u_x + v * u_y (nonlinear convection for u)
    conv_v = u * v_x + v * v_y  # u * v_x + v * v_y (nonlinear convection for v)
    
    continuity = u_x + v_y
    
    ns_u = u_xx + u_yy  # Second derivatives of u for Laplacian term
    ns_v = v_xx + v_yy  # Second derivatives of v for Laplacian term
    # Re = v*L/nu ===> v = Re * nu / L = 1/L = 1/5
    ReynoldsNum = 1.0/100.0
    # Navier-Stokes PDEs for u and v components
    # eq_u = u_t + conv_u + p_x - ns_u/ReynoldsNum
    # eq_v = v_t + conv_v + p_y - ns_v/ReynoldsNum
    eq_u = conv_u + p_x - ReynoldsNum * ns_u
    eq_v = conv_v + p_y - ReynoldsNum * ns_v
    return [continuity, eq_u, eq_v]



# -------------------------
# PDE residuals (RANS + k-ω)
# -------------------------
# y = [u, v, p, k, w]
def pde_rans(x, y):
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


