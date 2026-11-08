import numpy as np
from numpy import cos, sin
from cartpole.config import L, G, max_accel, Kp_theta, Kd_theta, Kd_x, Kp_x
from cartpole.utils import wrap_angle

def update_mode(mode, x, theta, omega, energy):
    if mode == "reset":
        if abs(x) < 0.01 and abs(energy + G * (L/2)) < 0.0005:
            mode = "swingup"
        else:
            mode = "reset"

    elif mode == "swingup":
        theta_error_from_top = np.pi - abs(theta)
        if theta_error_from_top < np.deg2rad(25) and abs(x) < 0.45:
            mode = "balance"
        else:
            mode = "swingup"

    return mode

def reset(x, v, theta, omega, energy):
    a_centre = 5*(-x) - 3*v
    energy_error = -G*L/2 - energy
    a_calmdown = -1.05*np.sign(energy_error) * omega * cos(theta)

    a = np.clip(a_centre + a_calmdown, -max_accel, max_accel)
    return a

def swingup(x, v, theta, omega, energy):
    if abs(theta) <= 1e-3 and abs(omega) <= 1e-3:
        a = 0.1
    else:
        energy_error = G * (L/2) - energy;
        a_swing = -2*np.sign(energy_error) * omega * cos(theta)
        a_centre = -5*x - 0.45*v
        a = np.clip(a_swing + a_centre, -max_accel, max_accel)
    return a

def balance(x, v, theta, omega, target_x, target_theta):
    x_error = target_x - x
    dtheta = Kp_x*(x_error) + Kd_x*(-v)
    new_target_theta = wrap_angle(target_theta + dtheta)
    theta_error = wrap_angle(new_target_theta - theta)
    alpha_adjust = Kp_theta*(theta_error) + Kd_theta*(-omega)

    # Cart acceleration based on required control angular acceleration
    a = np.clip(-(2*L*alpha_adjust + 3*G*sin(theta))/(3*cos(theta)), -max_accel, max_accel)
    return a