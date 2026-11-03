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
dt = 0.01 # Timestep [s]
animation_interval = 30 # Animation interval [ms]
frame_skip = int((1.7 * animation_interval) / (dt * 1000)) # 1.7 is hard coded to make the animation run at a reasonable speed
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

# Initial conditions
x = np.random.rand() * 0.6 - 0.3
v = 0
a = 0
theta = np.random.rand() * 2*np.pi
omega = 0
mode = "reset"

# Target position
target_theta = np.deg2rad(180)
target_x = 0
target_E = G * (L/2)

settle_condition = int(1 / dt) # 1 second of stability required to be considered settled

#----------------------- Functions --------------------------

def update_mode(mode, x, theta, omega, energy):
    if mode == "reset":
        if abs(x) < 0.01 and abs(energy + G * (L/2)) < 0.0006:
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

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def reset(x, v, theta, omega, energy):
    a_centre = 5*(-x) - 3*v
    energy_error = -G*L/2 - energy
    a_calmdown = -np.sign(energy_error) * omega * cos(theta)

    a = np.clip(a_centre + a_calmdown, -max_accel, max_accel)
    return a

def swingup(x, v, theta, omega, energy):
    if abs(theta) <= 1e-3 and abs(omega) <= 1e-3:
        a = 0.1
    else:
        energy_error = G * (L/2) - energy;
        a_swing = -np.sign(energy_error) * omega * cos(theta)
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

def init_anim():
    cart.set_xy(([], []))
    rod.set_data([], [])

    theta_line.set_data([], [])
    x_line.set_data([], [])
    accel_line.set_data([], [])
    energy_line.set_data([], [])

    info_text.set_text("")
    status_text.set_text("")

    return (cart, rod, info_text, status_text, theta_line, x_line, accel_line, energy_line)

def animate(i):
    i = i * frame_skip

    x = x_history[i]
    theta = theta_history[i]

    cart.set_xy((x - 0.04, -0.02))

    rod.set_data([x, x + L * sin(theta)], [0, -L * cos(theta)])

    # Update status text.
    info_text.set_text(
        f"Time: {t[i]:.2f} s\n"
        f"Mode: {mode_history[i]}\n"
        f"Cart position: {x:.3f} m\n"
        f"Angle: {np.rad2deg(theta):.1f}°\n"
    )
    status_text.set_text(
        f"Settled: {settled_history[i]}"
    )
    if settled_history[i] == True:
        status_text.set_color("green")

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

    energy_line.set_data(
        t[:i + 1],
        energy_history[:i + 1]
    )

    return (cart, rod, info_text, status_text, theta_line, x_line, accel_line, energy_line)

def run_simulation(x, v, theta, omega, a, mode):
    print("Starting simulation...")

    history = np.empty((num_steps, 6)) # 6 columns: x, v, theta, omega, a, energy
    mode_history = np.empty(num_steps, dtype=object)
    settled_history = np.empty(num_steps, dtype=bool)
    success = False

    for i in range(num_steps):

        k_energy = (1/6) * L**2 * omega**2
        gp_energy = G * -(L/2) * cos(theta)
        energy = k_energy + gp_energy

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

#----------------------- Main code --------------------------

success, settle_time, history, mode_history, settled_history = run_simulation(x, v, a, theta, omega, mode)
print(f"Success: {success}, Settling time: {settle_time:.2f} s")

print("Starting animation...")

x_history = history[:, 0]
v_history = history[:, 1]
theta_history = history[:, 2]
omega_history = history[:, 3]
a_history = history[:, 4]
energy_history = history[:, 5]

theta_history_deg = np.rad2deg(theta_history)
theta_error_history = wrap_angle(np.pi - theta_history)
theta_error_history_deg = np.rad2deg(theta_error_history)

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(
    6, 2,
    figure=fig,
    width_ratios=[2, 1],
    height_ratios=[1, 1, 1, 1, 1, 1],
    hspace=1.2,
    wspace=0.5
)
ax_anim = fig.add_subplot(gs[0:4, 0])
ax_theta = fig.add_subplot(gs[0:2, 1])
ax_x = fig.add_subplot(gs[2:4, 1])
ax_accel = fig.add_subplot(gs[4:6, 0])
ax_energy = fig.add_subplot(gs[4:6, 1])

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

info_text = ax_anim.text(
    0.03, 0.97, "",
    transform=ax_anim.transAxes,
    verticalalignment="top"
)
status_text = ax_anim.text(
    0.5, 0.97, "",
    transform=ax_anim.transAxes,
    verticalalignment="top",
    horizontalalignment="center",
    fontweight="bold",
    fontsize="large"
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
ax_accel.grid(True)

accel_line, = ax_accel.plot(
    [], [],
    linewidth=2,
    label="Control acceleration"
)

ax_accel.legend(loc="upper right")

# Setup energy plot
ax_energy.set_title("Pendulum Energy (J)")
ax_energy.set_xlabel("Time (s)")
ax_energy.set_ylabel("Energy (J)")
ax_energy.set_xlim(0, t_stop)
ax_energy.set_ylim(-2, 2)
ax_energy.grid(True)

energy_line, = ax_energy.plot(
    [], [],
    linewidth=2,
    label="Energy"
)

ax_energy.axhline(
    G*L/2, 
    linestyle="--",
    color="black",
    alpha=0.6, # transparency
    label="Upright target"
)

ax_energy.legend(loc="upper right")

ani = animation.FuncAnimation(
    fig, 
    animate, 
    frames=(len(t)-1)//frame_skip, 
    init_func=init_anim,
    interval=animation_interval, 
    blit=True
)
plt.show()