"""Interactive double pendulum simulation.

This script uses Lagrangian-derived equations of motion for a planar,
frictionless double pendulum. It provides sliders to adjust initial
angles and angular velocities and restart the simulation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# Physical constants
G = 9.81
M1 = 1.0
M2 = 1.0

# Lengths are mutable so the UI sliders can update them at runtime.
L1 = 1.0
L2 = 1.0

# Damping (joint friction) coefficients, updated by UI sliders.
C1 = 0.02
C2 = 0.02


def derivatives(state: np.ndarray, _t: float) -> np.ndarray:
    """Return time-derivative vector for the double pendulum state.

    state = [theta1, omega1, theta2, omega2]
    where theta is angle from the vertical and omega is angular velocity.

    Equations follow from Lagrangian mechanics for two rigid rods with
    point masses M1 and M2 at lengths L1 and L2.
    """

    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1

    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) ** 2
    den2 = (L2 / L1) * den1

    domega1 = (
        M2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
        + M2 * G * np.sin(theta2) * np.cos(delta)
        + M2 * L2 * omega2**2 * np.sin(delta)
        - (M1 + M2) * G * np.sin(theta1)
    ) / den1

    domega2 = (
        -M2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
        + (M1 + M2) * G * np.sin(theta1) * np.cos(delta)
        - (M1 + M2) * L1 * omega1**2 * np.sin(delta)
        - (M1 + M2) * G * np.sin(theta2)
    ) / den2

    # Simple linear damping to mimic joint friction / air losses.
    domega1 -= C1 * omega1
    domega2 -= C2 * omega2

    return np.array([omega1, domega1, omega2, domega2], dtype=float)


def rk4_step(state: np.ndarray, t: float, dt: float) -> np.ndarray:
    """One Runge-Kutta (RK4) integration step."""

    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def positions_from_state(state: np.ndarray) -> tuple[float, float, float, float]:
    """Convert angular coordinates to cartesian positions."""

    theta1, _, theta2, _ = state
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2


def run_simulation() -> None:
    """Launch interactive matplotlib simulation."""

    base_dt = 0.005
    t = 0.0

    # Defaults (degrees for user-facing sliders)
    default_theta1 = 120.0
    default_theta2 = -20.0
    default_omega1 = 0.0
    default_omega2 = 0.0
    default_l1 = 1.0
    default_l2 = 1.0
    default_c1 = 0.02
    default_c2 = 0.02
    default_speed = 1.0

    state = np.array(
        [
            np.radians(default_theta1),
            np.radians(default_omega1),
            np.radians(default_theta2),
            np.radians(default_omega2),
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.2, bottom=0.60)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_title("Double Pendulum (Lagrangian Dynamics)")
    ax.grid(True, alpha=0.2)

    (rod_line,) = ax.plot([], [], "o-", lw=2.5, markersize=8)
    (trace_line,) = ax.plot([], [], "-", lw=1.0, alpha=0.6)

    trace_x: list[float] = []
    trace_y: list[float] = []

    def redraw_from_state(current_state: np.ndarray) -> None:
        x1, y1, x2, y2 = positions_from_state(current_state)
        rod_line.set_data([0, x1, x2], [0, y1, y2])
        trace_line.set_data(trace_x, trace_y)

    # Sliders
    slider_color = "#ececec"
    ax_theta1 = plt.axes([0.2, 0.44, 0.65, 0.03], facecolor=slider_color)
    ax_theta2 = plt.axes([0.2, 0.39, 0.65, 0.03], facecolor=slider_color)
    ax_omega1 = plt.axes([0.2, 0.34, 0.65, 0.03], facecolor=slider_color)
    ax_omega2 = plt.axes([0.2, 0.29, 0.65, 0.03], facecolor=slider_color)
    ax_l1 = plt.axes([0.2, 0.24, 0.65, 0.03], facecolor=slider_color)
    ax_l2 = plt.axes([0.2, 0.19, 0.65, 0.03], facecolor=slider_color)
    ax_c1 = plt.axes([0.2, 0.14, 0.65, 0.03], facecolor=slider_color)
    ax_c2 = plt.axes([0.2, 0.09, 0.65, 0.03], facecolor=slider_color)
    ax_speed = plt.axes([0.2, 0.04, 0.65, 0.03], facecolor=slider_color)

    s_theta1 = Slider(ax_theta1, "θ1 (deg)", -180.0, 180.0, valinit=default_theta1)
    s_theta2 = Slider(ax_theta2, "θ2 (deg)", -180.0, 180.0, valinit=default_theta2)
    s_omega1 = Slider(ax_omega1, "ω1 (deg/s)", -720.0, 720.0, valinit=default_omega1)
    s_omega2 = Slider(ax_omega2, "ω2 (deg/s)", -720.0, 720.0, valinit=default_omega2)
    s_l1 = Slider(ax_l1, "L1", 0.2, 2.0, valinit=default_l1)
    s_l2 = Slider(ax_l2, "L2", 0.2, 2.0, valinit=default_l2)
    s_c1 = Slider(ax_c1, "Damping 1", 0.0, 0.2, valinit=default_c1)
    s_c2 = Slider(ax_c2, "Damping 2", 0.0, 0.2, valinit=default_c2)
    s_speed = Slider(ax_speed, "Speed", 0.1, 3.0, valinit=default_speed)

    # Buttons
    ax_reset = plt.axes([0.2, 0.0, 0.18, 0.03])
    ax_pause = plt.axes([0.42, 0.0, 0.18, 0.03])

    btn_reset = Button(ax_reset, "Apply / Reset")
    btn_pause = Button(ax_pause, "Pause")

    paused = {"value": False}

    def reset_state(_event=None) -> None:
        nonlocal state, t
        global L1, L2, C1, C2
        t = 0.0
        L1 = float(s_l1.val)
        L2 = float(s_l2.val)
        C1 = float(s_c1.val)
        C2 = float(s_c2.val)
        max_r = L1 + L2 + 0.2
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)

        state = np.array(
            [
                np.radians(s_theta1.val),
                np.radians(s_omega1.val),
                np.radians(s_theta2.val),
                np.radians(s_omega2.val),
            ],
            dtype=float,
        )
        trace_x.clear()
        trace_y.clear()
        redraw_from_state(state)
        fig.canvas.draw_idle()

    def toggle_pause(_event=None) -> None:
        paused["value"] = not paused["value"]
        btn_pause.label.set_text("Resume" if paused["value"] else "Pause")

    btn_reset.on_clicked(reset_state)
    btn_pause.on_clicked(toggle_pause)

    def animate(_frame: int):
        nonlocal state, t
        if not paused["value"]:
            sim_dt = base_dt * float(s_speed.val)
            substeps = max(1, int(np.ceil(sim_dt / 0.005)))
            dt_step = sim_dt / substeps
            for _ in range(substeps):
                state = rk4_step(state, t, dt_step)
                t += dt_step

            x1, y1, x2, y2 = positions_from_state(state)
            trace_x.append(x2)
            trace_y.append(y2)
            if len(trace_x) > 1500:
                del trace_x[:1]
                del trace_y[:1]
            rod_line.set_data([0, x1, x2], [0, y1, y2])
            trace_line.set_data(trace_x, trace_y)

        return rod_line, trace_line

    reset_state()
    # Keep a reference to the animation object; otherwise some backends
    # garbage-collect it and nothing is drawn/updated.
    animation = FuncAnimation(fig, animate, interval=16, blit=False, cache_frame_data=False)
    fig._animation = animation
    plt.show()


if __name__ == "__main__":
    run_simulation()
