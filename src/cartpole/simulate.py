import numpy as np
from cartpole.config import num_steps, settle_condition, target_theta, target_x, t
from cartpole.utils import wrap_angle
from cartpole.controllers import update_mode, reset, swingup, balance
from cartpole.dynamics import update_physics, pendulum_energy

def run_simulation(x, v, a, theta, omega, mode):
    print("Starting simulation...")

    history = np.empty((num_steps, 6)) # 6 columns: x, v, theta, omega, a, energy
    mode_history = np.empty(num_steps, dtype=object)
    settled_history = np.empty(num_steps, dtype=bool)
    settle_time = -1
    success = False

    for i in range(num_steps):
        energy = pendulum_energy(theta, omega)

        mode = update_mode(mode, x, theta, omega, energy)

        if mode == "reset":
            a = reset(x, v, theta, omega, energy)
        elif mode == "swingup":
            a = swingup(x, v, theta, omega, energy)
        elif mode == "balance":
            a = balance(x, v, theta, omega, target_x, target_theta)

        x, v, a, theta, omega = update_physics(x, v, a, theta, omega)

        history[i] = [x, v, theta, omega, a, energy]
        mode_history[i] = mode

    x_history = history[:, 0]
    theta_history = history[:, 2]

    theta_error_history = wrap_angle(np.pi - theta_history)
    theta_error_history_deg = np.rad2deg(theta_error_history)

    for i in range(settle_condition, num_steps):
        if (abs(x_history[i-settle_condition:i]) < 0.03).all() and (abs(theta_error_history_deg[i-settle_condition:i]) < 3).all():
            settled_history[i:] = True
            settle_time = t[i]
            success = True
            break        
        else:
            settled_history[i] = False

    print("Simulation complete.")
    return success, settle_time, history, mode_history, settled_history