from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cartpole.simulate import run_simulation
from cartpole.animate import run_animation

# Initial conditions
x = np.random.rand() * 0.6 - 0.3
v = 0
a = 0
theta = np.random.rand() * 2*np.pi
omega = 0
mode = "reset"

#----------------------- Main code --------------------------

success, settle_time, history, mode_history, settled_history = run_simulation(x, v, a, theta, omega, mode)
print(f"Success: {success}, Settling time: {settle_time:.2f} s")

run_animation(history, mode_history, settled_history)