import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from enum import Enum

# --------------------
# Config & constants
# --------------------
@dataclass
class Config:
    OUTPUT_DIR: str = "analysis_graphs"

class Propulsion(Enum):
    FIN = "Fin"
    FLAGELLA = "Flagella"

# Environment presets
ENVIRONMENTS = {
    "Water": {"rho": 1000.0, "mu": 0.001},
    "Honey": {"rho": 1400.0, "mu": 10.0},
    "Air": {"rho": 1.2, "mu": 1.8e-5},
    "Oil": {"rho": 900.0, "mu": 0.1},
    "Cytoplasm": {"rho": 1100.0, "mu": 0.01},
}

# --------------------
# Physics helpers
# --------------------
def reynolds(rho: float, U: float, L: float, mu: float) -> float:
    return 0 if mu <= 0 or L <= 0 else rho * U * L / mu

def cd_blend(Re: float) -> float:
    if Re < 1:
        return 24 / max(Re, 1e-2)
    elif Re < 1e3:
        return 24/Re + 6/(1+math.sqrt(Re)) + 0.4
    else:
        return 0.44

def fin_eff(Re: float) -> float:
    x = math.log10(max(Re, 1e-6))
    t = 1 / (1 + math.exp(-(x - 0.5) * 2.5))
    return max(0.02, min(1.0, t))

def flag_eff(Re: float) -> float:
    x = math.log10(max(Re, 1e-6))
    t = 1 / (1 + math.exp((x - 0.0) * 2.5))
    return max(0.05, min(1.2, t * 1.2))

# --------------------
# Simulator
# --------------------
class Simulator:
    def __init__(self):
        self.results = []

    def simulate(self, L, freq, rho, mu, propulsion, steps=1000, dt=0.001):
        vel_x = 0.0
        for _ in range(steps):
            Re = reynolds(rho, abs(vel_x), L, mu)

            if propulsion == Propulsion.FIN:
                eff = fin_eff(Re)
                amp = 0.2 * L
                thrust = 0.5 * rho * (freq * amp)**2 * L**2 * eff
            else:
                eff = flag_eff(Re)
                pitch = 0.2 * L
                thrust = 6 * math.pi * mu * (L/2) * (freq * pitch) * eff

            r = L/2
            A = math.pi * r**2
            Cd = cd_blend(max(Re, 1e-6))
            drag = 6*math.pi*mu*r*abs(vel_x) + 0.5*rho*Cd*A*vel_x**2
            mass = max(1e-12, rho*(4/3)*math.pi*r**3)

            acc = (thrust - drag)/mass
            vel_x = max(0, min(vel_x + acc*dt, 10.0))

        return {
            "velocity": vel_x,
            "reynolds": reynolds(rho, vel_x, L, mu),
            "efficiency": eff,
            "thrust": thrust,
            "drag": drag,
            "power": thrust*vel_x,
            "L": L,
            "freq": freq,
            "rho": rho,
            "mu": mu,
            "propulsion": propulsion.value,
        }

    # -------- Parameter sweeps ----------
    def sweep_viscosity(self, base_L, base_freq, base_rho):
        mus = np.logspace(-6, 2, 50)
        for mu in mus:
            for prop in Propulsion:
                self.results.append(
                    self.simulate(base_L, base_freq, base_rho, mu, prop) |
                    {"analysis": "viscosity"}
                )

    def sweep_size(self, base_freq, base_rho, base_mu):
        sizes = np.logspace(-6, 0, 40)
        for L in sizes:
            for prop in Propulsion:
                self.results.append(
                    self.simulate(L, base_freq, base_rho, base_mu, prop) |
                    {"analysis": "size"}
                )

    def sweep_frequency(self, base_L, base_rho, base_mu):
        freqs = np.logspace(-1, 3, 40)
        for f in freqs:
            for prop in Propulsion:
                self.results.append(
                    self.simulate(base_L, f, base_rho, base_mu, prop) |
                    {"analysis": "frequency"}
                )

    def sweep_environment(self, base_L, base_freq):
        for env, vals in ENVIRONMENTS.items():
            for prop in Propulsion:
                self.results.append(
                    self.simulate(base_L, base_freq, vals["rho"], vals["mu"], prop) |
                    {"analysis": "environment", "environment": env}
                )

# --------------------
# Plotting
# --------------------
class Plotter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        plt.style.use("dark_background")
        plt.rcParams["figure.facecolor"] = "#0f1923"
        plt.rcParams["axes.facecolor"] = "#1a2332"

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    def plot_viscosity(self):
        data = self.df[self.df.analysis=="viscosity"].sort_values("mu")
        fig, ax = plt.subplots(figsize=(8,6))
        for prop, color in [("Fin","#00b4ff"),("Flagella","#ffb400")]:
            subset = data[data.propulsion==prop]
            ax.loglog(subset.mu, subset.velocity, "o-", color=color, label=prop)
        ax.set_xlabel("Viscosity (Pa·s)"); ax.set_ylabel("Speed (m/s)")
        ax.set_title("Speed vs Viscosity")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(f"{Config.OUTPUT_DIR}/viscosity.png", dpi=200)
        plt.close()

    def plot_size(self):
        data = self.df[self.df.analysis=="size"].sort_values("L")
        fig, ax = plt.subplots(figsize=(8,6))
        for prop, color in [("Fin","#00b4ff"),("Flagella","#ffb400")]:
            subset = data[data.propulsion==prop]
            ax.loglog(subset.L, subset.velocity, "o-", color=color, label=prop)
        ax.set_xlabel("Size (m)"); ax.set_ylabel("Speed (m/s)")
        ax.set_title("Speed vs Size")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(f"{Config.OUTPUT_DIR}/size.png", dpi=200)
        plt.close()

    def plot_frequency(self):
        data = self.df[self.df.analysis=="frequency"].sort_values("freq")
        fig, ax = plt.subplots(figsize=(8,6))
        for prop, color in [("Fin","#00b4ff"),("Flagella","#ffb400")]:
            subset = data[data.propulsion==prop]
            ax.semilogx(subset.freq, subset.velocity, "o-", color=color, label=prop)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Speed (m/s)")
        ax.set_title("Speed vs Frequency")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(f"{Config.OUTPUT_DIR}/frequency.png", dpi=200)
        plt.close()

    def plot_environment(self):
        data = self.df[self.df.analysis=="environment"]
        if data.empty: return
        fig, ax = plt.subplots(figsize=(8,6))
        envs = data.environment.unique()
        bar_w = 0.35
        for i, prop in enumerate(["Fin","Flagella"]):
            vals = [data[(data.environment==e)&(data.propulsion==prop)].velocity.mean() for e in envs]
            ax.bar(np.arange(len(envs))+i*bar_w, vals, width=bar_w, label=prop)
        ax.set_xticks(np.arange(len(envs))+bar_w/2); ax.set_xticklabels(envs, rotation=45)
        ax.set_ylabel("Speed (m/s)"); ax.set_title("Environment Comparison")
        ax.legend(); plt.tight_layout()
        plt.savefig(f"{Config.OUTPUT_DIR}/environment.png", dpi=200)
        plt.close()

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    sim = Simulator()
    sim.sweep_viscosity(base_L=5e-6, base_freq=10, base_rho=1000)
    sim.sweep_size(base_freq=10, base_rho=1000, base_mu=0.001)
    sim.sweep_frequency(base_L=5e-6, base_rho=1000, base_mu=0.001)
    sim.sweep_environment(base_L=5e-6, base_freq=10)

    df = pd.DataFrame(sim.results)
    plotter = Plotter(df)
    plotter.plot_viscosity()
    plotter.plot_size()
    plotter.plot_frequency()
    plotter.plot_environment()
    print("✅ Graphs saved to 'analysis_graphs/'")
