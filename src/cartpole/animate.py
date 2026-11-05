import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import numpy as np
from numpy import cos, sin

from cartpole.config import G, L, dt, track_length, track_buffer, t_stop, max_accel, t
from cartpole.utils import wrap_angle

animation_interval = 30 # Animation interval [ms]
frame_skip = int((1.7 * animation_interval) / (dt * 1000)) # 1.7 is hard coded to make the animation run at a reasonable speed

def init_anim(cart, rod, info_text, status_text, theta_line, x_line, accel_line, energy_line):
    cart.set_xy(([], []))
    rod.set_data([], [])

    theta_line.set_data([], [])
    x_line.set_data([], [])
    accel_line.set_data([], [])
    energy_line.set_data([], [])

    info_text.set_text("")
    status_text.set_text("")
    status_text.set_color("black")

    return (cart, rod, info_text, status_text, theta_line, x_line, accel_line, energy_line)

def animate(i, cart, rod, info_text, status_text, theta_line, x_line, accel_line, energy_line, 
            x_history, theta_history, t, mode_history, settled_history, theta_error_history_deg, a_history, energy_history):
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



def run_animation(history, mode_history, settled_history):
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

    ax_anim.set_xlim(-(track_length / 2 + track_buffer), track_length / 2 + track_buffer)
    ax_anim.set_ylim(-L - track_buffer, L + track_buffer)
    ax_anim.set_aspect("equal")
    ax_anim.set_title("Cart-Pole Animation")
    ax_anim.set_xlabel("Cart Position (m)")
    ax_anim.set_ylabel("Height (m)")
    ax_anim.grid(True)

    track = ax_anim.plot(
        [-track_length/2, track_length/2], 
        [0, 0], 
        color="black", 
        lw=2, marker="|", markersize=10
    )
    ax_anim.axvline(-track_length / 2, linestyle="--", alpha=0.5)
    ax_anim.axvline(track_length / 2, linestyle="--", alpha=0.5)

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
    ax_x.set_ylim(-track_length/2 - 0.05, track_length/2 + 0.05)
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
        init_func=lambda: init_anim(
            cart, rod, info_text, status_text, theta_line, x_line, accel_line, energy_line
        ),
        fargs=(
            cart, rod, info_text, status_text, theta_line, x_line, accel_line, energy_line,
            x_history, theta_history, t, mode_history, settled_history,
            theta_error_history_deg, a_history, energy_history
        ),
        interval=animation_interval, 
        blit=True
    )
    plt.show()