import numpy as np
from cartpole.simulate import run_simulation
from cartpole.animate import run_animation

# Initial conditions
x = np.random.rand() * 0.6 - 0.3
v = 0
a = 0
theta = np.random.rand() * 2*np.pi - np.pi
omega = 0
mode = "reset"

#----------------------- Main code --------------------------

success, settle_time, history, mode_history, settled_history = run_simulation(
    x=x,
    v=v, 
    a=a, 
    theta=theta, 
    omega=omega, 
    mode=mode
)
print(f"x: {x:.3f}, theta: {np.rad2deg(theta):.1f}, Success: {success}, Settling time: {settle_time:.2f} s")

run_animation(history, mode_history, settled_history)