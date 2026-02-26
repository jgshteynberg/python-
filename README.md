# Double Pendulum — Super Simple Guide

You only need **2 commands**.

## Step 1) Install stuff (one time)

Copy/paste this:

```bash
python3 -m pip install -r requirements.txt
```

## Step 2) Start the simulator

Copy/paste this:

```bash
python3 double_pendulum_sim.py
```

A window opens with the pendulum.

---

## What buttons/sliders do

- `θ1`, `θ2` = starting angles
- `ω1`, `ω2` = starting speeds
- `L1`, `L2` = pendulum lengths
- `Damping 1`, `Damping 2` = friction/air-loss feel (higher = settles faster)
- `Speed` = simulation speed multiplier
- **Apply / Reset** = restart using slider values
- **Pause** = pause/resume

---

## If command says `python3: command not found`

Try:

```bash
python -m pip install -r requirements.txt
python double_pendulum_sim.py
```

---

## If command says `No module named ...`

Run install again:

```bash
python3 -m pip install -r requirements.txt
```

---

## One-command option

You can also run this helper script:

```bash
./run.sh
```

It installs requirements and starts the app.
