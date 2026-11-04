import numpy as np
from numpy import cos, sin
from cartpole.utils import wrap_angle
from cartpole.config import L, G, dt, B, m

def update_physics(x, v, a, theta, omega):
    v = v + a * dt
    x = x + v * dt

    if x <= -m/2 or x >= m/2:
        x = np.clip(x, -m/2, m/2)
        a = -v / dt
        v = 0

    alpha = -3/(2*L)*(a*cos(theta) + G*sin(theta)) - B*omega
    omega = omega + alpha * dt
    theta = theta + omega * dt
    theta = wrap_angle(theta)

    return x, v, a, theta, omega

def pendulum_energy(theta, omega):
    k_energy = (1/6) * L**2 * omega**2
    gp_energy = G * -(L/2) * cos(theta)
    
    energy = k_energy + gp_energy

    return energy