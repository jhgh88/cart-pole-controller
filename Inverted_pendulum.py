import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

#----------------------- Parameters --------------------------

# Physical constants
L = 0.3 # Pendulum length [m]
G = 9.81 # Gravity [m/s^2]
dt = 0.1 # Timestep [s]
B = 0.7 # Damping coefficient
max_accel = 10 # Maximum acceleration [m/s^2]
m = 1.0 # Length of track [m]
n = 0.1 # Buffer [m]

t_stop = 20.0 # Total simulation duration [s]
num_steps = int(t_stop / dt)
t = np.arange(0, num_steps + 1) * dt

# PID
Kp_theta = 140
Kd_theta = 15
Kp_x = -0.35
Kd_x = -0.18

# Initial conditions
theta = np.deg2rad(20)
omega = 0
x = np.random.rand()*0.6 - 0.3
v = 0
a = 0
energy = 0
gpe = 0
mode = "reset"

# Target position
target_theta = np.deg2rad(180)
target_x = 0
target_E = G * (L/2)

states = np.empty((num_steps, 5)) # 5 columns: x, v, theta, omega, a
mode_history = np.empty(num_steps, dtype=object)
settled_history = np.zeros(num_steps, dtype=bool)
energy_history = np.empty(num_steps)

#----------------------- Functions --------------------------

def update_mode(mode, x, theta, omega, energy):
    if mode == "reset":
        if abs(x) < 0.02 and abs(energy + G * (L/2)) < 0.0005:
            mode = "swingup"
        else:
            mode = "reset"

    elif mode == "swingup":
        theta_error_from_top = np.pi - abs(theta)
        if theta_error_from_top < np.deg2rad(20):
            mode = "balance"
        else:
            mode = "swingup"

    return mode

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def reset(x, v, theta, omega):
    a = 2*(-x) - 2*v
    return a

def swingup(x, v, theta, omega, energy):
    if abs(theta) <= 1e-3 and abs(omega) <= 1e-3:
        a = 0.01
    else:
        energy_error = G * (L/2) - energy;
        alpha_adjust = 2.5 * max(energy_error, 0.1*G*L) * omega * cos(theta)
        a_swing = -(2*L) / (3*cos(theta)) * alpha_adjust
        a_centre = -5*x - 0.5*v
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

def update_physics(x, v, a, theta, omega):
    v = v + a * dt
    x = x + v * dt
    print(f"t: {t[i]:.1f}, mode: {mode}, x: {x:.2f}, v: {v:.2f}, a: {a:.2f}")

    if x <= -m/2 or x >= m/2:
        x = np.clip(x, -m/2, m/2)
        a = -v / dt
        v = 0

    alpha = -3/(2*L)*(a*cos(theta) + G*sin(theta)) - B*omega
    omega = omega + alpha * dt
    theta = theta + omega * dt
    theta = wrap_angle(theta)

    return x, v, a, theta, omega

def init_anim():
    cart.set_xy(([], []))
    rod.set_data([], [])

    theta_line.set_data([], [])
    x_line.set_data([], [])
    accel_line.set_data([], [])

    status_text.set_text("")

    return (cart, rod, status_text, theta_line, x_line, accel_line)

def animate(i):
    x = x_history[i]
    theta = theta_history[i]

    cart.set_xy((x - 0.04, -0.02))

    rod.set_data([x, x + L * sin(theta)], [0, -L * cos(theta)])

    # Update status text.
    status_text.set_text(
        f"Time: {t[i]:.1f} s\n"
        f"Mode: {mode_history[i]}\n"
        f"Cart position: {x:.3f} m\n"
        f"Angle: {np.rad2deg(theta):.1f}°\n"
        f"Settled: {settled_history[i]}"
    )

    # Update graph histories.
    # i+1 includes the current frame.
    theta_line.set_data(
        t[:i + 1],
        theta_error_history_deg[:i + 1]
    )

    x_line.set_data(
        t[:i + 1],
        x_history[:i + 1]
    )

    accel_line.set_data(
        t[:i + 1],
        a_history[:i + 1]
    )

    return (cart, rod, status_text, theta_line, x_line, accel_line)

#----------------------- Main code --------------------------

print("Starting simulation...")

for i in range(num_steps):

    k_energy = (1/6) * L**2 * omega**2
    gp_energy = G * -(L/2) * cos(theta)
    energy = k_energy + gp_energy

    mode = update_mode(mode, x, theta, omega, energy)

    if mode == "reset":
        a = reset(x, v, theta, omega)
    elif mode == "swingup":
        a = swingup(x, v, theta, omega, energy)
    elif mode == "balance":
        a = balance(x, v, theta, omega, target_x, target_theta)

    x, v, a, theta, omega = update_physics(x, v, a, theta, omega)

    states[i] = [x, v, theta, omega, a]
    mode_history[i] = mode
    energy_history[i] = energy

x_history = states[:, 0]
v_history = states[:, 1]
theta_history = states[:, 2]
omega_history = states[:, 3]
a_history = states[:, 4]

theta_history_deg = np.rad2deg(theta_history)
theta_error_history = wrap_angle(np.pi - theta_history)
theta_error_history_deg = np.rad2deg(theta_error_history)

print("Simulation complete.")

print("Starting animation...")

fig = plt.figure(figsize=(12, 7))
gs = GridSpec(
    3, 2,
    figure=fig,
    width_ratios=[1, 1],
    height_ratios=[1, 1, 1],
    hspace=0.6,
    wspace=0.3
)
ax_anim = fig.add_subplot(gs[:, 0])
ax_theta = fig.add_subplot(gs[0, 1])
ax_x = fig.add_subplot(gs[1, 1])
ax_accel = fig.add_subplot(gs[2, 1])

fig.suptitle("Cart-pole Simulation", fontsize=16, fontweight="bold")

ax_anim.set_xlim(-(m / 2 + n), m / 2 + n)
ax_anim.set_ylim(-L - n, L + n)
ax_anim.set_aspect("equal")
ax_anim.set_title("Cart-Pole Animation")
ax_anim.set_xlabel("Cart Position (m)")
ax_anim.set_ylabel("Height (m)")
ax_anim.grid(True)

track = ax_anim.plot(
    [-m/2, m/2], 
    [0, 0], 
    color="black", 
    lw=2, marker="|", markersize=10
)
ax_anim.axvline(-m / 2, linestyle="--", alpha=0.5)
ax_anim.axvline(m / 2, linestyle="--", alpha=0.5)

cart = Rectangle(
    (x_history[0] - 0.04, -0.02), 
    0.08, 
    0.04, 
    fc="grey", 
    ec="black", 
    zorder=2
)
ax_anim.add_patch(cart)

rod, = ax_anim.plot([], [], 'o-', lw=2, color="blue")

status_text = ax_anim.text(
    0.03, 0.97, "",
    transform=ax_anim.transAxes,
    verticalalignment="top"
)

# Setup angle error plot
ax_theta.set_title("Pendulum Angle Error (deg)")
ax_theta.set_xlabel("Time (s)")
ax_theta.set_ylabel("Error Angle (deg)")
ax_theta.set_xlim(0, t_stop)
ax_theta.set_ylim(-180, 180)
ax_theta.grid(True)

theta_line, = ax_theta.plot(
    [], [],
    linewidth=2,
    label="Angle error"
)

ax_theta.axhline(
    0, 
    linestyle="--",
    color="black",
    alpha=0.6, # transparency
    label="Upright target"
)

ax_theta.legend(loc="upper right")

# Setup cart position plot
ax_x.set_title("Cart Position (m)")
ax_x.set_xlabel("Time (s)")
ax_x.set_ylabel("Position (m)")
ax_x.set_xlim(0, t_stop)
ax_x.set_ylim(-m/2 - 0.05, m/2 + 0.05)
ax_x.grid(True)

x_line, = ax_x.plot(
    [], [],
    linewidth=2,
    label="Cart position"
)

ax_x.axhline(
    0, 
    linestyle="--",
    color="black",
    alpha=0.6, # transparency
    label="Target"
)

ax_x.legend(loc="upper right")

# Setup acceleration graph

ax_accel.set_title("Pendulum Acceleration (m/s^2)")
ax_accel.set_xlabel("Time (s)")
ax_accel.set_ylabel("Acceleration (m/s^2)")
ax_accel.set_xlim(0, t_stop)
ax_accel.set_ylim(-max_accel, max_accel)
ax_x.grid(True)

accel_line, = ax_accel.plot(
    [], [],
    linewidth=2,
    label="Control acceleration"
)

ax_accel.legend(loc="upper right")

ani = animation.FuncAnimation(
    fig, 
    animate, 
    frames=len(t)-1, 
    init_func=init_anim,
    interval=dt*1000, 
    blit=True
)
plt.show()