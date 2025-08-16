"""
Minimal Locomotion Demo – Bacteria vs Fish (Vertical Layout)
+ DATA/GRAPH GENERATOR

Additions in this version
-------------------------
• Press **P** to generate a big bundle of CSV datasets and PNG graphs under ./outputs
  (for BOTH scenarios and ALL environments), without leaving the game.
• Or run:  `python locomotion_demo.py --batch`  to generate everything headlessly.
• Sweeps cover viscosity, frequency, size (L), and combine to produce heatmaps.
• Graphs include: speed vs viscosity (fin vs flagella), Re vs viscosity, speed vs frequency,
  speed vs size, efficiency vs Re, and 2D heatmaps of speed and Re over (L,f).
• All raw data saved to CSV for your own plotting.

Install:  pip install pygame matplotlib numpy pandas
Run UI:   python locomotion_demo.py
Batch:    python locomotion_demo.py --batch

Keys in UI:  1/2 scenarios, Z/X/C/V/B environments, Q/A freq, W/S size,
             F labels, H help, **P generate plots**, ESC quit.
"""

import argparse
import os
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict

import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------
# Config & constants
# --------------------
@dataclass
class Config:
    WIDTH: int = 1200
    HEIGHT: int = 700
    FPS: int = 60
    FONT_SIZE: int = 18

    BG: Tuple[int, int, int] = (15, 25, 35)
    GRID: Tuple[int, int, int] = (38, 48, 60)
    TEXT: Tuple[int, int, int] = (225, 230, 235)
    HI: Tuple[int, int, int] = (255, 255, 120)

    FIN: Tuple[int, int, int] = (0, 180, 255)
    FLAG: Tuple[int, int, int] = (255, 180, 0)

    # Visual scaling so small things are still visible
    SPEED_VISUAL: float = 250.0


class Propulsion(Enum):
    FIN = "Fin"
    FLAGELLA = "Flagella"


# Environment presets
ENVIRONMENTS: Dict[str, Dict] = {
    "Water": {"rho": 1000.0, "mu": 0.001, "color": (100, 150, 255)},
    "Honey": {"rho": 1400.0, "mu": 10.0, "color": (255, 200, 50)},
    "Air": {"rho": 1.2, "mu": 1.8e-5, "color": (200, 220, 255)},
    "Oil": {"rho": 900.0, "mu": 0.1, "color": (120, 80, 40)},
    "Cytoplasm": {"rho": 1100.0, "mu": 0.01, "color": (150, 255, 150)},
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Physics (simple but didactic)
# --------------------

def reynolds(rho: float, U: float, L: float, mu: float) -> float:
    if mu <= 0 or L <= 0:
        return 0.0
    return rho * U * L / mu


def cd_blend(Re: float) -> float:
    """Blend low-Re (Stokes) and quadratic drag. Returns a nominal Cd for the quadratic term.
    (We also include a Stokes linear term separately.)"""
    if Re < 1:
        return 24 / max(Re, 1e-2)
    elif Re < 1e3:
        return 24 / Re + 6 / (1 + math.sqrt(Re)) + 0.4
    else:
        return 0.44


def fin_eff(Re: float) -> float:
    """Fins ineffective at low Re, effective at high Re."""
    x = math.log10(max(Re, 1e-6))
    t = 1 / (1 + math.exp(-(x - 0.5) * 2.5))
    return max(0.02, min(1.0, t))


def flag_eff(Re: float) -> float:
    """Flagella great at low Re, fade at high Re."""
    x = math.log10(max(Re, 1e-6))
    t = 1 / (1 + math.exp((x - 0.0) * 2.5))
    return max(0.05, min(1.2, t * 1.2))


# --------------------
# Creature (simulation carrier)
# --------------------
class Creature:
    def __init__(self, label: str, x: float, y: float, color: Tuple[int, int, int], L: float, freq: float, propulsion: Propulsion):
        self.label = label
        self.start_x = x
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0.0, 0.0)
        self.color = color
        self.L = L  # characteristic length (m)
        self.freq = freq  # Hz
        self.amp = max(1e-6, 0.2 * L)  # stroke amplitude (m)
        self.pitch = max(1e-6, 0.2 * L)  # flagellar pitch proxy (m/rev)
        self.propulsion = propulsion
        self.anim_t = 0.0
        self.Re = 0.0
        self.speed = 0.0
        self.eff = 0.0

    def radius_px(self) -> int:
        if self.L < 1e-4:
            return 12
        if self.L < 1e-2:
            return 20
        if self.L < 1e-1:
            return 32
        if self.L < 5e-1:
            return 45
        return 58

    def update(self, dt: float, rho: float, mu: float):
        U = self.vel.x
        Re = reynolds(rho, abs(U), self.L, mu)

        # Thrust models
        if self.propulsion == Propulsion.FIN:
            eff = fin_eff(Re)
            thrust = 0.5 * rho * (self.freq * self.amp) ** 2 * self.L ** 2 * eff
        else:
            eff = flag_eff(Re)
            thrust = 6 * math.pi * mu * (self.L / 2) * (self.freq * self.pitch) * eff

        # Drag: linear (Stokes) + quadratic
        r = self.L / 2
        A = math.pi * r * r
        Cd = cd_blend(max(Re, 1e-6))
        drag_linear = 6 * math.pi * mu * r * abs(U)
        drag_quad = 0.5 * rho * Cd * A * U * U
        drag = drag_linear + drag_quad

        mass = max(1e-9, rho * (4 / 3) * math.pi * r ** 3)
        net = thrust - drag
        ax = net / mass
        self.vel.x += ax * dt
        self.vel.x = max(0.0, min(self.vel.x, 3.0))
        self.pos.x += self.vel.x * dt * Config.SPEED_VISUAL

        self.Re = reynolds(rho, abs(self.vel.x), self.L, mu)
        self.speed = self.vel.x
        self.eff = eff

        if self.pos.x > Config.WIDTH + 60:
            self.pos.x = self.start_x

        self.anim_t += dt

    def draw(self, screen: pygame.Surface, show_labels: bool = True):
        rpx = self.radius_px()
        pos = (int(self.pos.x), int(self.pos.y))
        pygame.draw.circle(screen, self.color, pos, rpx)

        if self.propulsion == Propulsion.FIN:
            tail = [
                (pos[0] - rpx, pos[1]),
                (pos[0] - rpx - 2 * rpx, pos[1] - int(rpx * 0.6 * math.sin(self.anim_t * 6))),
                (pos[0] - rpx - 2 * rpx, pos[1] + int(rpx * 0.6 * math.sin(self.anim_t * 6))),
            ]
            pygame.draw.polygon(screen, self.color, tail, 2)
        else:
            pts: List[Tuple[int, int]] = []
            length = 3 * rpx
            segs = 24
            for i in range(segs + 1):
                p = i / segs
                x = pos[0] - rpx - int(p * length)
                y = pos[1] + int(0.35 * rpx * math.sin(10 * p + self.anim_t * 12))
                pts.append((x, y))
            pygame.draw.lines(screen, self.color, False, pts, 2)

        if show_labels:
            lbl = f"{self.label} — {self.propulsion.value}"
            font = pygame.font.SysFont("Arial", 16, bold=True)
            surf = font.render(lbl, True, self.color)
            screen.blit(surf, (pos[0] - surf.get_width() // 2, pos[1] - rpx - 25))


# --------------------
# UI helpers
# --------------------
class UI:
    def __init__(self, font_main: pygame.font.Font, font_small: pygame.font.Font):
        self.font = font_main
        self.small = font_small

    def draw_grid(self, screen: pygame.Surface):
        step = 50
        for x in range(0, Config.WIDTH, step):
            pygame.draw.line(screen, Config.GRID, (x, 0), (x, Config.HEIGHT), 1)
        for y in range(0, Config.HEIGHT, step):
            pygame.draw.line(screen, Config.GRID, (0, y), (Config.WIDTH, y), 1)

    def panel(self, screen: pygame.Surface, lines: List[str], x: int, y: int, bg_color: Tuple[int, int, int] = (10, 15, 25)):
        pads = 12
        spacing = 3
        line_surfs = [self.font.render(line, True, Config.TEXT) for line in lines if line.strip()]
        if not line_surfs:
            return
        w = max((s.get_width() for s in line_surfs), default=0) + pads * 2
        h = sum(s.get_height() for s in line_surfs) + spacing * (len(line_surfs) - 1) + pads * 2
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        if len(bg_color) >= 3:
            fill_color = (bg_color[0], bg_color[1], bg_color[2], 210)
        else:
            fill_color = (10, 15, 25, 210)
        surf.fill(fill_color)
        pygame.draw.rect(surf, (255, 255, 255, 30), surf.get_rect(), 1)
        screen.blit(surf, (x, y))
        cy = y + pads
        for s in line_surfs:
            screen.blit(s, (x + pads, cy))
            cy += s.get_height() + spacing

    def help_overlay(self, screen: pygame.Surface):
        text = [
            "LOCOMOTION DEMO - VERTICAL LAYOUT",
            "",
            "Scenarios: 1=Bacteria  2=Fish",
            "",
            "Both Creatures Controls:",
            "Q/A: Frequency   W/S: Size",
            "",
            "Environments:",
            "Z=Water  X=Honey  C=Air  V=Oil  B=Cytoplasm",
            "",
            "Other: F=toggle labels  H=close help  P=generate plots",
        ]
        pads = 20
        lines = [self.font.render(t, True, Config.TEXT) for t in text]
        w = max(s.get_width() for s in lines) + pads * 2
        h = sum(s.get_height() for s in lines) + pads * 2 + 6 * (len(lines) - 1)
        x = (Config.WIDTH - w) // 2
        y = (Config.HEIGHT - h) // 2
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((5, 10, 15, 240))
        pygame.draw.rect(box, Config.HI, box.get_rect(), 3)
        screen.blit(box, (x, y))
        cy = y + pads
        for s in lines:
            screen.blit(s, (x + pads, cy))
            cy += s.get_height() + 6

    def draw_static_attributes(self, screen: pygame.Surface, creature: Creature, x: int, y: int):
        lines = [
            f"{creature.propulsion.value.upper()}",
            f"Size: {creature.L:.2e} m",
            f"Freq: {creature.freq:.1f} Hz",
            f"Re: {creature.Re:.2f}",
            f"Speed: {creature.speed:.3f} m/s",
            f"Efficiency: {creature.eff:.2f}",
        ]
        bg_color = creature.color[:3] if len(creature.color) >= 3 else (10, 15, 25)
        self.panel(screen, lines, x, y, bg_color)


# --------------------
# Data & plotting utilities (NEW)
# --------------------
class Simulator:
    """A light integrator that reuses Creature.update without opening a window."""

    def __init__(self, dt: float = 1 / 240.0, warmup: float = 3.0, sample: float = 1.0):
        self.dt = dt
        self.warmup = warmup  # seconds to reach near steady-state
        self.sample = sample  # seconds to average speed

    def steady_speed(self, L: float, freq: float, rho: float, mu: float, propulsion: Propulsion) -> Tuple[float, float, float]:
        # minimal pygame-free vector stand-in
        class Dummy:
            x = 0.0
            y = 0.0

        dummy = pygame.Vector2(0.0, 0.0)
        c = Creature("S", 0.0, 0.0, (255, 255, 255), L, freq, propulsion)
        c.pos = dummy  # we won't use drawing
        # warmup
        t = 0.0
        while t < self.warmup:
            c.update(self.dt, rho, mu)
            t += self.dt
        # sample
        v_sum = 0.0
        n = 0
        t = 0.0
        while t < self.sample:
            c.update(self.dt, rho, mu)
            v_sum += c.speed
            n += 1
            t += self.dt
        v = v_sum / max(1, n)
        Re = reynolds(rho, abs(v), L, mu)
        eff = fin_eff(Re) if propulsion == Propulsion.FIN else flag_eff(Re)
        return v, Re, eff


def save_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def save_fig(name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


class DataGenerator:
    def __init__(self):
        self.sim = Simulator()

    # ------------- 1) Speed vs viscosity (two lines) -------------
    def speed_vs_viscosity(self, scenario: str, L: float, freq: float, rhos: List[float], mus: np.ndarray):
        rows = []
        for rho in rhos:
            for mu in mus:
                for prop in (Propulsion.FIN, Propulsion.FLAGELLA):
                    v, Re, eff = self.sim.steady_speed(L, freq, rho, float(mu), prop)
                    rows.append({
                        "scenario": scenario,
                        "rho": rho,
                        "mu": mu,
                        "propulsion": prop.value,
                        "speed": v,
                        "Re": Re,
                        "eff": eff,
                        "L": L,
                        "freq": freq,
                    })
        df = pd.DataFrame(rows)
        name = f"speed_vs_viscosity_{scenario}"
        csv_path = save_csv(df, name)
        # plot
        plt.figure()
        for prop in ("Fin", "Flagella"):
            sub = df[df["propulsion"] == prop].groupby("mu")["speed"].mean().reset_index()
            plt.plot(sub["mu"], sub["speed"], label=prop)
        plt.xscale("log")
        plt.xlabel("Viscosity μ (Pa·s)")
        plt.ylabel("Steady speed (m/s)")
        plt.title(f"Speed vs Viscosity — {scenario}")
        plt.legend()
        png_path = save_fig(name)
        return csv_path, png_path

    # ------------- 2) Speed vs frequency -------------
    def speed_vs_frequency(self, scenario: str, L: float, rho: float, mu_list: List[float], freqs: np.ndarray):
        rows = []
        for mu in mu_list:
            for freq in freqs:
                for prop in (Propulsion.FIN, Propulsion.FLAGELLA):
                    v, Re, eff = self.sim.steady_speed(L, float(freq), rho, mu, prop)
                    rows.append({
                        "scenario": scenario, "mu": mu, "propulsion": prop.value, "speed": v, "Re": Re, "eff": eff, "L": L, "freq": float(freq),
                    })
        df = pd.DataFrame(rows)
        name = f"speed_vs_frequency_{scenario}"
        csv_path = save_csv(df, name)
        plt.figure()
        for prop in ("Fin", "Flagella"):
            for mu in mu_list:
                sub = df[(df["propulsion"] == prop) & (df["mu"] == mu)]
                plt.plot(sub["freq"], sub["speed"], label=f"{prop} μ={mu}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Steady speed (m/s)")
        plt.title(f"Speed vs Frequency — {scenario}")
        plt.legend()
        png_path = save_fig(name)
        return csv_path, png_path

    # ------------- 3) Speed vs size L -------------
    def speed_vs_size(self, scenario: str, freq: float, rho: float, mu: float, Ls: np.ndarray):
        rows = []
        for L in Ls:
            for prop in (Propulsion.FIN, Propulsion.FLAGELLA):
                v, Re, eff = self.sim.steady_speed(float(L), freq, rho, mu, prop)
                rows.append({"scenario": scenario, "L": float(L), "propulsion": prop.value, "speed": v, "Re": Re, "eff": eff, "mu": mu, "rho": rho, "freq": freq})
        df = pd.DataFrame(rows)
        name = f"speed_vs_size_{scenario}"
        csv_path = save_csv(df, name)
        plt.figure()
        for prop in ("Fin", "Flagella"):
            sub = df[df["propulsion"] == prop].sort_values("L")
            plt.plot(sub["L"], sub["speed"], label=prop)
        plt.xscale("log")
        plt.xlabel("Size L (m)")
        plt.ylabel("Steady speed (m/s)")
        plt.title(f"Speed vs Size — {scenario}")
        plt.legend()
        png_path = save_fig(name)
        return csv_path, png_path

    # ------------- 4) Re vs viscosity (two lines) -------------
    def re_vs_viscosity(self, scenario: str, L: float, freq: float, rhos: List[float], mus: np.ndarray):
        rows = []
        for rho in rhos:
            for mu in mus:
                for prop in (Propulsion.FIN, Propulsion.FLAGELLA):
                    v, Re, eff = self.sim.steady_speed(L, freq, rho, float(mu), prop)
                    rows.append({"scenario": scenario, "rho": rho, "mu": mu, "propulsion": prop.value, "speed": v, "Re": Re})
        df = pd.DataFrame(rows)
        name = f"Re_vs_viscosity_{scenario}"
        csv_path = save_csv(df, name)
        plt.figure()
        for prop in ("Fin", "Flagella"):
            sub = df[df["propulsion"] == prop].groupby("mu")["Re"].mean().reset_index()
            plt.plot(sub["mu"], sub["Re"], label=prop)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Viscosity μ (Pa·s)")
        plt.ylabel("Reynolds number")
        plt.title(f"Re vs Viscosity — {scenario}")
        plt.legend()
        png_path = save_fig(name)
        return csv_path, png_path

    # ------------- 5) Efficiency curves vs Re -------------
    def efficiency_vs_Re(self):
        Re = np.logspace(-6, 4, 400)
        fin = [fin_eff(r) for r in Re]
        flag = [flag_eff(r) for r in Re]
        df = pd.DataFrame({"Re": Re, "Fin_eff": fin, "Flagella_eff": flag})
        name = "efficiency_vs_Re"
        csv_path = save_csv(df, name)
        plt.figure()
        plt.plot(Re, fin, label="Fin efficiency")
        plt.plot(Re, flag, label="Flagella efficiency")
        plt.xscale("log")
        plt.xlabel("Reynolds number")
        plt.ylabel("Efficiency (arb.)")
        plt.title("Efficiency vs Reynolds number")
        plt.legend()
        png_path = save_fig(name)
        return csv_path, png_path

    # ------------- 6) Heatmaps over (L, f) -------------
    def heatmap_over_L_f(self, scenario: str, rho: float, mu: float, Ls: np.ndarray, freqs: np.ndarray):
        for prop in (Propulsion.FIN, Propulsion.FLAGELLA):
            speed_grid = np.zeros((len(Ls), len(freqs)))
            Re_grid = np.zeros_like(speed_grid)
            for i, L in enumerate(Ls):
                for j, f in enumerate(freqs):
                    v, Re, _ = self.sim.steady_speed(float(L), float(f), rho, mu, prop)
                    speed_grid[i, j] = v
                    Re_grid[i, j] = Re
            # Save CSVs (long-form)
            rows = []
            for i, L in enumerate(Ls):
                for j, f in enumerate(freqs):
                    rows.append({"scenario": scenario, "propulsion": prop.value, "L": float(L), "freq": float(f), "speed": speed_grid[i, j], "Re": Re_grid[i, j], "mu": mu, "rho": rho})
            df = pd.DataFrame(rows)
            base = f"heatmap_{prop.value.lower()}_{scenario}_mu{mu}"
            save_csv(df, base)

            # Plot Speed heatmap
            plt.figure()
            plt.imshow(speed_grid, aspect="auto", origin="lower", extent=[freqs.min(), freqs.max(), Ls.min(), Ls.max()])
            plt.yscale("log")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Size L (m)")
            plt.title(f"Speed heatmap — {prop.value} — {scenario} (μ={mu})")
            cbar = plt.colorbar()
            cbar.set_label("Speed (m/s)")
            save_fig(base + "_speed")

            # Plot Re heatmap
            plt.figure()
            plt.imshow(Re_grid, aspect="auto", origin="lower", extent=[freqs.min(), freqs.max(), Ls.min(), Ls.max()])
            plt.yscale("log")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Size L (m)")
            plt.title(f"Re heatmap — {prop.value} — {scenario} (μ={mu})")
            cbar = plt.colorbar()
            cbar.set_label("Re")
            save_fig(base + "_Re")

    # ------------- master pipeline -------------
    def generate_all(self):
        # Scenarios and canonical sizes/freqs
        scenarios = {
            "bacteria": {"L": 5e-6, "freq": 100.0},
            "fish": {"L": 0.1, "freq": 4.0},
        }
        # Compose density values from presets
        rhos = [ENVIRONMENTS[name]["rho"] for name in ENVIRONMENTS]
        mus = np.logspace(-5, 1.3, 36)  # ~air to honey
        # Frequency sweeps (bacteria higher range)
        freqs_bac = np.linspace(5, 300, 40)
        freqs_fish = np.linspace(0.5, 10, 40)
        # Size sweeps across bacteria->fish
        Ls_all = np.logspace(-6, -0.5, 40)

        # 0) Efficiency vs Re (global)
        self.efficiency_vs_Re()

        for scen_name, params in scenarios.items():
            L0 = params["L"]
            f0 = params["freq"]

            # 1) Speed vs viscosity
            self.speed_vs_viscosity(scen_name, L0, f0, rhos=rhos, mus=mus)

            # 2) Speed vs frequency at a few μ
            mu_list = [ENVIRONMENTS["Air"]["mu"], ENVIRONMENTS["Water"]["mu"], ENVIRONMENTS["Honey"]["mu"]]
            rho_water = ENVIRONMENTS["Water"]["rho"]
            freqs = freqs_bac if scen_name == "bacteria" else freqs_fish
            self.speed_vs_frequency(scen_name, L0, rho_water, mu_list, freqs)

            # 3) Speed vs size at a canonical μ (water)
            self.speed_vs_size(scen_name, f0, rho_water, ENVIRONMENTS["Water"]["mu"], Ls_all)

            # 4) Re vs viscosity
            self.re_vs_viscosity(scen_name, L0, f0, rhos=rhos, mus=mus)

            # 5) Heatmaps in water for each propulsion
            Ls_heat = np.logspace(math.log10(L0) - 1.0, math.log10(L0) + 1.0, 36)
            freqs_heat = np.logspace(-1 if scen_name == "fish" else 0, 2.5 if scen_name == "bacteria" else 1.2, 48)
            self.heatmap_over_L_f(scen_name, rho_water, ENVIRONMENTS["Water"]["mu"], Ls_heat, freqs_heat)

        print(f"All datasets and plots saved under: {OUTPUT_DIR}")


# --------------------
# Game
# --------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
        pygame.display.set_caption("Locomotion Demo – Vertical Layout with Environment Control + Data")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", Config.FONT_SIZE, bold=True)
        self.small = pygame.font.SysFont("Arial", 16)
        self.ui = UI(self.small, self.small)

        # Environment
        self.current_env = "Water"
        self.rho = ENVIRONMENTS[self.current_env]["rho"]
        self.mu = ENVIRONMENTS[self.current_env]["mu"]

        self.scenario = "bacteria"  # or "fish"
        self.show_labels = True
        self._build_scenario()
        self.show_help = False

    def _build_scenario(self):
        cx = Config.WIDTH * 0.5
        cy_top = Config.HEIGHT * 0.3
        cy_bottom = Config.HEIGHT * 0.7

        if self.scenario == "bacteria":
            L = 5e-6
            f = 100.0
            self.top = Creature("Bacterium", cx, cy_top, Config.FIN, L, f, Propulsion.FIN)
            self.bottom = Creature("Bacterium", cx, cy_bottom, Config.FLAG, L, f, Propulsion.FLAGELLA)
        else:
            L = 0.1
            f = 4.0
            self.top = Creature("Fish", cx, cy_top, Config.FIN, L, f, Propulsion.FIN)
            self.bottom = Creature("Fish", cx, cy_bottom, Config.FLAG, L, f, Propulsion.FLAGELLA)

    def _set_environment(self, env_name: str):
        if env_name in ENVIRONMENTS:
            self.current_env = env_name
            env = ENVIRONMENTS[env_name]
            self.rho = env["rho"]
            self.mu = env["mu"]

    def _generate_plots(self):
        gen = DataGenerator()
        gen.generate_all()

    def run(self):
        while True:
            dt = self.clock.tick(Config.FPS) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    if e.key == pygame.K_1:
                        self.scenario = "bacteria"; self._build_scenario()
                    if e.key == pygame.K_2:
                        self.scenario = "fish"; self._build_scenario()
                    if e.key == pygame.K_f:
                        self.show_labels = not self.show_labels
                    if e.key == pygame.K_h:
                        self.show_help = not self.show_help
                    # Environment keys
                    if e.key == pygame.K_z:
                        self._set_environment("Water")
                    if e.key == pygame.K_x:
                        self._set_environment("Honey")
                    if e.key == pygame.K_c:
                        self._set_environment("Air")
                    if e.key == pygame.K_v:
                        self._set_environment("Oil")
                    if e.key == pygame.K_b:
                        self._set_environment("Cytoplasm")
                    # NEW: generate plots
                    if e.key == pygame.K_p:
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        # quick toast-style overlay via console
                        print("Generating datasets & plots...")
                        self._generate_plots()
                        print("Done. Check ./outputs")

            keys = pygame.key.get_pressed()

            # Both creatures' frequency controls (Q/A)
            if keys[pygame.K_q]:
                self.top.freq = min(1000, self.top.freq * 1.02)
                self.bottom.freq = min(1000, self.bottom.freq * 1.02)
            if keys[pygame.K_a]:
                self.top.freq = max(0.1, self.top.freq * 0.98)
                self.bottom.freq = max(0.1, self.bottom.freq * 0.98)

            # Both creatures' size controls (W/S)
            if keys[pygame.K_w]:
                self.top.L = min(2.0, self.top.L * 1.02)
                self.top.amp = max(1e-6, 0.2 * self.top.L)
                self.top.pitch = max(1e-6, 0.2 * self.top.L)
                self.bottom.L = min(2.0, self.bottom.L * 1.02)
                self.bottom.amp = max(1e-6, 0.2 * self.bottom.L)
                self.bottom.pitch = max(1e-6, 0.2 * self.bottom.L)
            if keys[pygame.K_s]:
                self.top.L = max(1e-6, self.top.L * 0.98)
                self.top.amp = max(1e-6, 0.2 * self.top.L)
                self.top.pitch = max(1e-6, 0.2 * self.top.L)
                self.bottom.L = max(1e-6, self.bottom.L * 0.98)
                self.bottom.amp = max(1e-6, 0.2 * self.bottom.L)
                self.bottom.pitch = max(1e-6, 0.2 * self.bottom.L)

            self.top.update(dt, self.rho, self.mu)
            self.bottom.update(dt, self.rho, self.mu)

            # Draw
            env_color = ENVIRONMENTS[self.current_env]["color"]
            bg_tinted = tuple(int(Config.BG[i] * 0.8 + env_color[i] * 0.2) for i in range(3))
            self.screen.fill(bg_tinted)

            self.ui.draw_grid(self.screen)
            self.top.draw(self.screen, self.show_labels)
            self.bottom.draw(self.screen, self.show_labels)

            # Static attribute displays on the left
            self.ui.draw_static_attributes(self.screen, self.top, 20, int(self.top.pos.y) - 60)
            self.ui.draw_static_attributes(self.screen, self.bottom, 20, int(self.bottom.pos.y) - 60)

            # Environment info panel (top-right)
            env_lines = [
                f"Environment: {self.current_env}",
                f"Density: {self.rho:.2g} kg/m³",
                f"Viscosity: {self.mu:.2g} Pa·s",
                "",
                f"Scenario: {'Bacteria' if self.scenario == 'bacteria' else 'Fish'}",
                "",
                "Expected Winner:",
                f"• Bacteria: {'Flagella' if self.scenario == 'bacteria' else 'N/A'}",
                f"• Fish: {'Fins' if self.scenario == 'fish' else 'N/A'}",
                "",
                "P: generate CSVs & plots → ./outputs",
            ]
            self.ui.panel(self.screen, env_lines, Config.WIDTH - 360, 20)

            # Controls reminder (bottom-right)
            control_lines = [
                "Controls:",
                "Q/A=frequency W/S=size",
                "Env: Z/X/C/V/B",
                "1/2=scenario F=labels H=help",
                "P=export plots",
            ]
            self.ui.panel(self.screen, control_lines, Config.WIDTH - 240, Config.HEIGHT - 160)

            if self.show_help:
                self.ui.help_overlay(self.screen)

            pygame.display.flip()


# --------------------
# Entry point
# --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Generate all datasets and plots headlessly, no UI")
    args = parser.parse_args()

    if args.batch:
        gen = DataGenerator()
        gen.generate_all()
        return

    Game().run()


if __name__ == "__main__":
    main()