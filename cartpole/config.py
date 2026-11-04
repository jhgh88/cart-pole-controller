import numpy as np

# Physical constants
L = 0.3 # Pendulum length [m]
G = 9.81 # Gravity [m/s^2]
dt = 0.01 # Timestep [s]
B = 0.1 # Damping coefficient
max_accel = 10 # Maximum acceleration [m/s^2]
m = 1.0 # Length of track [m]
n = 0.1 # Buffer [m]
t_stop = 10.0 # Total simulation duration [s]
num_steps = int(t_stop / dt)
t = np.arange(0, num_steps + 1) * dt

# PID
Kp_theta = 150
Kd_theta = 20
Kp_x = -0.35
Kd_x = -0.3

# Target position
target_theta = np.deg2rad(180)
target_x = 0

settle_condition = int(1 / dt) # 1 second of stability required to be considered settled