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
L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0


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

    dt = 0.01
    t = 0.0

    # Defaults (degrees for user-facing sliders)
    default_theta1 = 120.0
    default_theta2 = -20.0
    default_omega1 = 0.0
    default_omega2 = 0.0

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
    plt.subplots_adjust(left=0.2, bottom=0.35)
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
    ax_theta1 = plt.axes([0.2, 0.24, 0.65, 0.03], facecolor=slider_color)
    ax_theta2 = plt.axes([0.2, 0.19, 0.65, 0.03], facecolor=slider_color)
    ax_omega1 = plt.axes([0.2, 0.14, 0.65, 0.03], facecolor=slider_color)
    ax_omega2 = plt.axes([0.2, 0.09, 0.65, 0.03], facecolor=slider_color)

    s_theta1 = Slider(ax_theta1, "θ1 (deg)", -180.0, 180.0, valinit=default_theta1)
    s_theta2 = Slider(ax_theta2, "θ2 (deg)", -180.0, 180.0, valinit=default_theta2)
    s_omega1 = Slider(ax_omega1, "ω1 (deg/s)", -720.0, 720.0, valinit=default_omega1)
    s_omega2 = Slider(ax_omega2, "ω2 (deg/s)", -720.0, 720.0, valinit=default_omega2)

    # Buttons
    ax_reset = plt.axes([0.2, 0.02, 0.18, 0.05])
    ax_pause = plt.axes([0.42, 0.02, 0.18, 0.05])

    btn_reset = Button(ax_reset, "Apply / Reset")
    btn_pause = Button(ax_pause, "Pause")

    paused = {"value": False}

    def reset_state(_event=None) -> None:
        nonlocal state, t
        t = 0.0
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
            state = rk4_step(state, t, dt)
            t += dt

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
    animation = FuncAnimation(fig, animate, interval=16, blit=True, cache_frame_data=False)
    fig._animation = animation
    plt.show()


if __name__ == "__main__":
    run_simulation()
