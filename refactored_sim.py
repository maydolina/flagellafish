
"""
Refactored Bio/Fluid Motility Simulator & Plotter
-------------------------------------------------
A clean, fully self-contained rewrite of the original ~1800-line script.
Goals:
  - Deterministic, meaningful graphs (no scribbles)
  - Sorted data in every plot
  - Clear separation of concerns (simulation, sweeps, plotting, UI)
  - Optional pygame UI (keeps your original keybindings concept)
  - Single-file for portability; no external assets needed

Usage
-----
1) Headless (recommended) to generate all graphs:
     python refactored_sim.py --all-graphs

2) Interactive (pygame) to tweak parameters and run one-shot sims:
     python refactored_sim.py --ui

3) Run specific sweep and plot:
     python refactored_sim.py --sweep viscosity --save

Output
------
- CSV:   ./out/data_*.csv
- Plots: ./out/plots/*.png

Notes
-----
- Uses only standard libs + numpy, pandas, matplotlib, pygame (optional).
- Matplotlib style: dark background; grid; markers; sorting before line plots.
- Physics model preserves your intent but stabilizes numerics.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lazy import pygame (only if --ui)
_HAVE_PYGAME = False
try:
    import pygame  # type: ignore
    _HAVE_PYGAME = True
except Exception:
    _HAVE_PYGAME = False

# ----------------------
# Configuration & Colors
# ----------------------
@dataclass
class Config:
    out_dir: str = "out"
    plots_dir: str = os.path.join("out", "plots")
    csv_dir: str = os.path.join("out", "csv")
    # Visualization
    face_color: str = "#0f1923"
    axes_color: str = "#1a2332"

CFG = Config()

# Ensure output dirs
os.makedirs(CFG.out_dir, exist_ok=True)
os.makedirs(CFG.plots_dir, exist_ok=True)
os.makedirs(CFG.csv_dir, exist_ok=True)

# ---------------
# Domain Entities
# ---------------
class Propulsion(str, Enum):
    FIN = "Fin"
    FLAGELLA = "Flagella"

ENVIRONMENTS: Dict[str, Dict[str, float]] = {
    "Water": {"rho": 1000.0, "mu": 1.0e-3},
    "Honey": {"rho": 1400.0, "mu": 10.0},
    "Air": {"rho": 1.2, "mu": 1.8e-5},
    "Oil": {"rho": 900.0, "mu": 0.1},
    "Cytoplasm": {"rho": 1100.0, "mu": 0.01},
}

# ---------------
# Physics Helpers
# ---------------
def reynolds(rho: float, U: float, L: float, mu: float) -> float:
    """Compute Reynolds number (clamped for stability)."""
    if mu <= 0 or L <= 0:
        return 0.0
    return max(0.0, rho * U * L / mu)

def cd_blend(Re: float) -> float:
    """Composite drag coefficient correlation."""
    Re = max(Re, 1e-9)
    if Re < 1:
        return 24 / Re               # Stokes
    elif Re < 1e3:
        return 24/Re + 6/(1+math.sqrt(Re)) + 0.4  # Schiller-Naumann blend
    else:
        return 0.44                  # High-Re flat-plate-ish

def fin_efficiency(Re: float) -> float:
    """Heuristic fin efficiency vs Re (smooth, bounded)."""
    x = math.log10(max(Re, 1e-9))
    t = 1 / (1 + math.exp(-(x - 0.5) * 2.5))
    return min(1.0, max(0.05, t))

def flagella_efficiency(Re: float) -> float:
    """Heuristic flagella efficiency vs Re (opposite trend at low Re)."""
    x = math.log10(max(Re, 1e-9))
    t = 1 / (1 + math.exp((x - 0.0) * 2.5))
    return min(1.2, max(0.05, t * 1.2))

# -----------------
# Core Simulation
# -----------------
@dataclass
class State:
    U: float = 0.0  # forward velocity [m/s]

@dataclass
class Params:
    L: float             # characteristic length [m]
    freq: float          # stroke/beat frequency [Hz]
    rho: float           # density [kg/m3]
    mu: float            # viscosity [Pa·s]
    propulsion: Propulsion

class Integrator:
    """Simple explicit Euler with damping/clamps for stability."""
    def __init__(self, dt: float = 1e-3, max_U: float = 20.0):
        self.dt = dt
        self.max_U = max_U

    def step(self, state: State, acc: float) -> State:
        U = state.U + acc * self.dt
        U = max(0.0, min(self.max_U, U))
        return State(U=U)

class Simulator:
    def __init__(self, dt: float = 1e-3, steps: int = 1500):
        self.dt = dt
        self.steps = steps
        self.integrator = Integrator(dt=dt)

    def _thrust(self, p: Params, Re: float, U: float) -> float:
        """Propulsion-dependent thrust model (simple but stable)."""
        L = p.L
        rho, mu, f = p.rho, p.mu, p.freq

        if p.propulsion == Propulsion.FIN:
            eff = fin_efficiency(Re)
            amp = 0.2 * L
            # Dynamic pressure * fin area * eff
            return 0.5 * rho * (f * amp)**2 * L**2 * eff
        else:
            eff = flagella_efficiency(Re)
            r = L/2
            # Low-Re inspired: 6*pi*mu*r*v (use f*stroke_length as velocity scale)
            stroke = 0.2 * L
            return 6 * math.pi * mu * r * (f * stroke) * eff

    def _drag(self, p: Params, U: float, Re: float) -> float:
        """Blend Stokes + quadratic drag."""
        r = max(1e-12, p.L / 2)
        A = math.pi * r**2
        Cd = cd_blend(Re)
        stokes = 6 * math.pi * p.mu * r * U
        quad = 0.5 * p.rho * Cd * A * U**2
        return stokes + quad

    def run(self, p: Params) -> Dict[str, float]:
        state = State(U=0.0)
        mass = max(1e-12, p.rho * (4/3) * math.pi * (p.L/2)**3)

        for _ in range(self.steps):
            Re = reynolds(p.rho, state.U, p.L, p.mu)
            T = self._thrust(p, Re, state.U)
            D = self._drag(p, state.U, Re)
            a = (T - D) / mass
            state = self.integrator.step(state, a)

        Re = reynolds(p.rho, state.U, p.L, p.mu)
        T = self._thrust(p, Re, state.U)
        D = self._drag(p, state.U, Re)
        P = T * state.U
        eff = fin_efficiency(Re) if p.propulsion == Propulsion.FIN else flagella_efficiency(Re)

        return {
            "velocity": state.U,
            "reynolds": Re,
            "thrust": T,
            "drag": D,
            "power": P,
            "efficiency": eff,
            "L": p.L,
            "freq": p.freq,
            "rho": p.rho,
            "mu": p.mu,
            "propulsion": p.propulsion.value,
        }

# -----------------
# Sweep Generation
# -----------------
class Sweeps:
    def __init__(self, sim: Simulator):
        self.sim = sim

    def viscosity(self, L: float, freq: float, rho: float,
                  mus: Optional[np.ndarray] = None) -> pd.DataFrame:
        mus = mus if mus is not None else np.logspace(-6, 2, 60)
        rows = []
        for mu in mus:
            for prop in Propulsion:
                p = Params(L=L, freq=freq, rho=rho, mu=float(mu), propulsion=prop)
                out = self.sim.run(p)
                out["analysis"] = "viscosity"
                rows.append(out)
        df = pd.DataFrame(rows)
        return df.sort_values(["propulsion", "mu"])

    def size(self, freq: float, rho: float, mu: float,
             sizes: Optional[np.ndarray] = None) -> pd.DataFrame:
        sizes = sizes if sizes is not None else np.logspace(-6, 0, 60)
        rows = []
        for L in sizes:
            for prop in Propulsion:
                p = Params(L=float(L), freq=freq, rho=rho, mu=mu, propulsion=prop)
                out = self.sim.run(p)
                out["analysis"] = "size"
                rows.append(out)
        df = pd.DataFrame(rows)
        return df.sort_values(["propulsion", "L"])

    def frequency(self, L: float, rho: float, mu: float,
                  freqs: Optional[np.ndarray] = None) -> pd.DataFrame:
        freqs = freqs if freqs is not None else np.logspace(-1, 3, 60)
        rows = []
        for f in freqs:
            for prop in Propulsion:
                p = Params(L=L, freq=float(f), rho=rho, mu=mu, propulsion=prop)
                out = self.sim.run(p)
                out["analysis"] = "frequency"
                rows.append(out)
        df = pd.DataFrame(rows)
        return df.sort_values(["propulsion", "freq"])

    def environment(self, L: float, freq: float) -> pd.DataFrame:
        rows = []
        for env, vals in ENVIRONMENTS.items():
            for prop in Propulsion:
                p = Params(L=L, freq=freq, rho=vals["rho"], mu=vals["mu"], propulsion=prop)
                out = self.sim.run(p)
                out["analysis"] = "environment"
                out["environment"] = env
                rows.append(out)
        df = pd.DataFrame(rows)
        return df.sort_values(["environment", "propulsion"])

# ---------
# Plotting
# ---------
class Plotter:
    def __init__(self):
        plt.style.use("dark_background")
        plt.rcParams["figure.facecolor"] = CFG.face_color
        plt.rcParams["axes.facecolor"] = CFG.axes_color

    def _save(self, name: str):
        path = os.path.join(CFG.plots_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"[plot] saved: {path}")

    def viscosity(self, df: pd.DataFrame):
        data = df[df.analysis == "viscosity"].copy()
        if data.empty: return
        data = data.sort_values(["propulsion", "mu"])
        fig, ax = plt.subplots(figsize=(8,6))
        for prop in [Propulsion.FIN.value, Propulsion.FLAGELLA.value]:
            sub = data[data.propulsion == prop]
            ax.loglog(sub["mu"], sub["velocity"], "o-", label=prop)
        ax.set_xlabel("Viscosity (Pa·s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Speed vs Viscosity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._save("viscosity.png")

    def size(self, df: pd.DataFrame):
        data = df[df.analysis == "size"].copy()
        if data.empty: return
        data = data.sort_values(["propulsion", "L"])
        fig, ax = plt.subplots(figsize=(8,6))
        for prop in [Propulsion.FIN.value, Propulsion.FLAGELLA.value]:
            sub = data[data.propulsion == prop]
            ax.loglog(sub["L"], sub["velocity"], "o-", label=prop)
        ax.set_xlabel("Size (m)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Speed vs Size")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._save("size.png")

    def frequency(self, df: pd.DataFrame):
        data = df[df.analysis == "frequency"].copy()
        if data.empty: return
        data = data.sort_values(["propulsion", "freq"])
        fig, ax = plt.subplots(figsize=(8,6))
        for prop in [Propulsion.FIN.value, Propulsion.FLAGELLA.value]:
            sub = data[data.propulsion == prop]
            ax.semilogx(sub["freq"], sub["velocity"], "o-", label=prop)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Speed vs Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._save("frequency.png")

    def environment(self, df: pd.DataFrame):
        data = df[df.analysis == "environment"].copy()
        if data.empty: return
        envs = list(data["environment"].unique())
        idx = np.arange(len(envs))
        width = 0.35
        fin_vals = [data[(data.environment==e) & (data.propulsion==Propulsion.FIN.value)]["velocity"].mean() for e in envs]
        fla_vals = [data[(data.environment==e) & (data.propulsion==Propulsion.FLAGELLA.value)]["velocity"].mean() for e in envs]

        fig, ax = plt.subplots(figsize=(9,6))
        ax.bar(idx, fin_vals, width, label="Fin")
        ax.bar(idx + width, fla_vals, width, label="Flagella")
        ax.set_xticks(idx + width/2)
        ax.set_xticklabels(envs, rotation=30)
        ax.set_ylabel("Mean Speed (m/s)")
        ax.set_title("Environment Comparison")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        self._save("environment.png")

# -----------------
# Pygame UI (Optional)
# -----------------
class UI:
    HELP_LINES = [
        "Controls:",
        "Q/A = increase/decrease frequency",
        "W/S = increase/decrease size (L)",
        "E/D = increase/decrease viscosity (mu)",
        "R/F = increase/decrease density (rho)",
        "1/2 = switch propulsion (Fin/Flagella)",
        "SPACE = run one-shot sim",
        "G = generate all graphs",
        "ESC = quit",
    ]

    def __init__(self):
        if not _HAVE_PYGAME:
            raise RuntimeError("pygame not available. Install pygame or run headless.")
        pygame.init()
        self.size = (900, 600)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Motility Simulator (Refactor)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        # Default params
        self.L = 5e-6
        self.freq = 10.0
        self.mu = 1e-3
        self.rho = 1000.0
        self.prop = Propulsion.FIN

        self.sim = Simulator(dt=1e-3, steps=1200)
        self.sweeps = Sweeps(self.sim)
        self.plotter = Plotter()

    def draw_text(self, text: str, x: int, y: int, color=(220, 230, 255)):
        surf = self.font.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def run(self):
        running = True
        result_text = ""
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # run one-shot
                        out = self.sim.run(Params(L=self.L, freq=self.freq, rho=self.rho, mu=self.mu, propulsion=self.prop))
                        result_text = f"U={out['velocity']:.4g} m/s | Re={out['reynolds']:.3g} | Thrust={out['thrust']:.3g} N"
                    elif event.key == pygame.K_q:
                        self.freq *= 1.2
                    elif event.key == pygame.K_a:
                        self.freq /= 1.2
                    elif event.key == pygame.K_w:
                        self.L *= 1.2
                    elif event.key == pygame.K_s:
                        self.L /= 1.2
                    elif event.key == pygame.K_e:
                        self.mu *= 1.5
                    elif event.key == pygame.K_d:
                        self.mu /= 1.5
                    elif event.key == pygame.K_r:
                        self.rho *= 1.1
                    elif event.key == pygame.K_f:
                        self.rho /= 1.1
                    elif event.key == pygame.K_1:
                        self.prop = Propulsion.FIN
                    elif event.key == pygame.K_2:
                        self.prop = Propulsion.FLAGELLA
                    elif event.key == pygame.K_g:
                        # generate all graphs
                        self._generate_all_graphs()

            self.screen.fill((16, 24, 36))
            y = 20
            for line in self.HELP_LINES:
                self.draw_text(line, 20, y); y += 22
            y += 10
            self.draw_text(f"L (m): {self.L:.3e}", 20, y); y += 20
            self.draw_text(f"freq (Hz): {self.freq:.3g}", 20, y); y += 20
            self.draw_text(f"mu (Pa·s): {self.mu:.3g}", 20, y); y += 20
            self.draw_text(f"rho (kg/m^3): {self.rho:.3g}", 20, y); y += 20
            self.draw_text(f"propulsion: {self.prop.value}", 20, y); y += 20
            y += 10
            self.draw_text(result_text, 20, y, color=(140, 255, 170))

            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def _generate_all_graphs(self):
        df_v = self.sweeps.viscosity(L=self.L, freq=self.freq, rho=self.rho)
        df_s = self.sweeps.size(freq=self.freq, rho=self.rho, mu=self.mu)
        df_f = self.sweeps.frequency(L=self.L, rho=self.rho, mu=self.mu)
        df_e = self.sweeps.environment(L=self.L, freq=self.freq)

        df = pd.concat([df_v, df_s, df_f, df_e], ignore_index=True)
        df.to_csv(os.path.join(CFG.csv_dir, "data_all.csv"), index=False)

        self.plotter.viscosity(df)
        self.plotter.size(df)
        self.plotter.frequency(df)
        self.plotter.environment(df)

# -------------
# CLI Entrypoint
# -------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Refactored Motility Simulator")
    ap.add_argument("--ui", action="store_true", help="launch pygame UI")
    ap.add_argument("--all-graphs", action="store_true", help="run all sweeps and save all plots")
    ap.add_argument("--sweep", choices=["viscosity","size","frequency","environment"], help="run one sweep and save csv/plot")
    ap.add_argument("--save", action="store_true", help="save outputs for single sweep")
    args = ap.parse_args(argv)

    sim = Simulator(dt=1e-3, steps=1400)
    sweeps = Sweeps(sim)
    plotter = Plotter()

    if args.ui:
        if not _HAVE_PYGAME:
            print("pygame not installed. Try: pip install pygame")
            return 1
        UI().run()
        return 0

    if args.all-graphs:
        L, f, rho, mu = 5e-6, 10.0, 1000.0, 1e-3
        df_v = sweeps.viscosity(L=L, freq=f, rho=rho)
        df_s = sweeps.size(freq=f, rho=rho, mu=mu)
        df_f = sweeps.frequency(L=L, rho=rho, mu=mu)
        df_e = sweeps.environment(L=L, freq=f)
        df = pd.concat([df_v, df_s, df_f, df_e], ignore_index=True)
        df.to_csv(os.path.join(CFG.csv_dir, "data_all.csv"), index=False)
        plotter.viscosity(df)
        plotter.size(df)
        plotter.frequency(df)
        plotter.environment(df)
        print("[done] all graphs generated into ./out/plots")
        return 0

    if args.sweep:
        L, f, rho, mu = 5e-6, 10.0, 1000.0, 1e-3
        if args.sweep == "viscosity":
            df = sweeps.viscosity(L=L, freq=f, rho=rho)
            if args.save:
                df.to_csv(os.path.join(CFG.csv_dir, "data_viscosity.csv"), index=False)
            plotter.viscosity(df)
        elif args.sweep == "size":
            df = sweeps.size(freq=f, rho=rho, mu=mu)
            if args.save:
                df.to_csv(os.path.join(CFG.csv_dir, "data_size.csv"), index=False)
            plotter.size(df)
        elif args.sweep == "frequency":
            df = sweeps.frequency(L=L, rho=rho, mu=mu)
            if args.save:
                df.to_csv(os.path.join(CFG.csv_dir, "data_frequency.csv"), index=False)
            plotter.frequency(df)
        elif args.sweep == "environment":
            df = sweeps.environment(L=L, freq=f)
            if args.save:
                df.to_csv(os.path.join(CFG.csv_dir, "data_environment.csv"), index=False)
            plotter.environment(df)
        print(f"[done] sweep={args.sweep}")
        return 0

    # Default: print help
    ap.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main())
