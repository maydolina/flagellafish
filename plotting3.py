import pygame
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List
import json
import os


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

    SPEED_VISUAL_BACTERIA: float = 50000.0
    SPEED_VISUAL_FISH: float = 250.0


class Propulsion(Enum):
    FIN = "Fin"
    FLAGELLA = "Flagella"


# Environment presets
ENVIRONMENTS = {
    "Water": {"rho": 1000.0, "mu": 0.001, "color": (100, 150, 255)},
    "Honey": {"rho": 1400.0, "mu": 10.0, "color": (255, 200, 50)},
    "Air": {"rho": 1.2, "mu": 0.000018, "color": (200, 220, 255)},
    "Oil": {"rho": 900.0, "mu": 0.1, "color": (120, 80, 40)},
    "Cytoplasm": {"rho": 1100.0, "mu": 0.01, "color": (150, 255, 150)}
}


# --------------------
# Physics (simple)
# --------------------
def reynolds(rho: float, U: float, L: float, mu: float) -> float:
    if mu <= 0 or L <= 0:
        return 0.0
    return rho * U * L / mu


def cd_blend(Re: float) -> float:
    if Re < 1:
        return 24 / max(Re, 1e-2)
    elif Re < 1e3:
        return 24 / Re + 6 / (1 + math.sqrt(Re)) + 0.4
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
# Data Analysis System
# --------------------
class DataAnalyzer:
    def __init__(self):
        self.data_points = []

    def calculate_performance(self, L: float, freq: float, rho: float, mu: float, propulsion: Propulsion,
                              steps: int = 1000):
        """Simulate creature performance and return steady-state values"""
        # Initial conditions
        vel_x = 0.0
        dt = 0.001  # Small time step for numerical stability

        # Simulate until steady state
        for _ in range(steps):
            U = vel_x
            Re = reynolds(rho, abs(U), L, mu)

            if propulsion == Propulsion.FIN:
                eff = fin_eff(Re)
                amp = max(1e-6, 0.2 * L)
                thrust = 0.5 * rho * (freq * amp) ** 2 * L ** 2 * eff
            else:
                eff = flag_eff(Re)
                pitch = max(1e-6, 0.2 * L)
                thrust = 6 * math.pi * mu * (L / 2) * (freq * pitch) * eff

            r = L / 2
            A = math.pi * r * r
            Cd = cd_blend(max(Re, 1e-6))
            drag_linear = 6 * math.pi * mu * r * abs(U)
            drag_quad = 0.5 * rho * Cd * A * U * U
            drag = drag_linear + drag_quad

            mass = max(1e-9, rho * (4 / 3) * math.pi * r ** 3)
            net = thrust - drag
            ax = net / mass
            vel_x += ax * dt
            vel_x = max(0.0, min(vel_x, 10.0))  # Clamp velocity

        final_Re = reynolds(rho, abs(vel_x), L, mu)
        final_eff = fin_eff(final_Re) if propulsion == Propulsion.FIN else flag_eff(final_Re)

        return {
            'velocity': vel_x,
            'reynolds': final_Re,
            'efficiency': final_eff,
            'thrust': thrust,
            'drag': drag,
            'power': thrust * vel_x,
            'L': L,
            'freq': freq,
            'rho': rho,
            'mu': mu,
            'propulsion': propulsion.value
        }

    def generate_comprehensive_data(self):
        """Generate data for all analysis graphs"""
        print("Generating comprehensive analysis data...")

        # Parameter ranges for analysis
        viscosities = np.logspace(-6, 2, 50)  # 1e-6 to 100 Pa·s
        sizes_bacteria = np.logspace(-6, -4, 30)  # 1μm to 100μm
        sizes_fish = np.logspace(-3, 0, 30)  # 1mm to 1m
        frequencies = np.logspace(-1, 3, 40)  # 0.1 to 1000 Hz
        densities = np.logspace(0, 4, 30)  # 1 to 10000 kg/m³
        reynolds_range = np.logspace(-6, 6, 50)

        # Base parameters
        base_rho = 1000.0  # water density
        base_mu = 0.001  # water viscosity
        base_freq = 10.0
        base_L_bacteria = 5e-6
        base_L_fish = 0.1

        self.data_points = []

        # 1. Viscosity vs Speed analysis
        print("  Analyzing viscosity effects...")
        for mu in viscosities:
            for prop in [Propulsion.FIN, Propulsion.FLAGELLA]:
                # Bacteria scale
                data_bacteria = self.calculate_performance(base_L_bacteria, base_freq, base_rho, mu, prop)
                data_bacteria['scale'] = 'bacteria'
                self.data_points.append(data_bacteria)

                # Fish scale
                data_fish = self.calculate_performance(base_L_fish, base_freq, base_rho, mu, prop)
                data_fish['scale'] = 'fish'
                self.data_points.append(data_fish)

        # 2. Size vs Performance analysis
        print("  Analyzing size effects...")
        for L in np.concatenate([sizes_bacteria, sizes_fish]):
            scale = 'bacteria' if L < 1e-4 else 'fish'
            for prop in [Propulsion.FIN, Propulsion.FLAGELLA]:
                data = self.calculate_performance(L, base_freq, base_rho, base_mu, prop)
                data['scale'] = scale
                self.data_points.append(data)

        # 3. Frequency vs Performance analysis
        print("  Analyzing frequency effects...")
        for freq in frequencies:
            for prop in [Propulsion.FIN, Propulsion.FLAGELLA]:
                # Bacteria
                data_bacteria = self.calculate_performance(base_L_bacteria, freq, base_rho, base_mu, prop)
                data_bacteria['scale'] = 'bacteria'
                self.data_points.append(data_bacteria)

                # Fish
                data_fish = self.calculate_performance(base_L_fish, freq, base_rho, base_mu, prop)
                data_fish['scale'] = 'fish'
                self.data_points.append(data_fish)

        # 4. Density vs Performance analysis
        print("  Analyzing density effects...")
        for rho in densities:
            for prop in [Propulsion.FIN, Propulsion.FLAGELLA]:
                # Bacteria
                data_bacteria = self.calculate_performance(base_L_bacteria, base_freq, rho, base_mu, prop)
                data_bacteria['scale'] = 'bacteria'
                self.data_points.append(data_bacteria)

                # Fish
                data_fish = self.calculate_performance(base_L_fish, base_freq, rho, base_mu, prop)
                data_fish['scale'] = 'fish'
                self.data_points.append(data_fish)

        # 5. Environment-specific analysis
        print("  Analyzing environment-specific performance...")
        for env_name, env_data in ENVIRONMENTS.items():
            for prop in [Propulsion.FIN, Propulsion.FLAGELLA]:
                # Test different sizes in each environment
                for L in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                    scale = 'bacteria' if L < 1e-4 else 'fish'
                    data = self.calculate_performance(L, base_freq, env_data['rho'], env_data['mu'], prop)
                    data['scale'] = scale
                    data['environment'] = env_name
                    self.data_points.append(data)

        print(f"Generated {len(self.data_points)} data points for analysis")

    def create_all_graphs(self):
        """Create all analysis graphs"""
        if not self.data_points:
            self.generate_comprehensive_data()

        # Set up matplotlib style
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#0f1923'
        plt.rcParams['axes.facecolor'] = '#1a2332'

        # Create output directory
        if not os.path.exists('analysis_graphs'):
            os.makedirs('analysis_graphs')

        # Convert data to arrays for easier analysis
        import pandas as pd
        df = pd.DataFrame(self.data_points)

        # 1. Viscosity vs Speed
        self.plot_viscosity_vs_speed(df)

        # 2. Reynolds Number Analysis
        self.plot_reynolds_analysis(df)

        # 3. Size vs Performance
        self.plot_size_vs_performance(df)

        # 4. Frequency Analysis
        self.plot_frequency_analysis(df)

        # 5. Efficiency Comparisons
        self.plot_efficiency_analysis(df)

        # 6. Power Analysis
        self.plot_power_analysis(df)

        # 7. Environment Comparison
        self.plot_environment_analysis(df)

        # 8. Thrust vs Drag Analysis
        self.plot_thrust_drag_analysis(df)

        # 9. Performance Regime Maps
        self.plot_performance_regime_maps(df)

        # 10. Scaling Laws
        self.plot_scaling_laws(df)

        # 11. Optimal Frequency Analysis
        self.plot_optimal_frequency(df)

        # 12. Multi-parameter Analysis
        self.plot_multi_parameter_analysis(df)

        print("All graphs generated and saved to 'analysis_graphs' folder")

    def plot_viscosity_vs_speed(self, df):
        """Plot speed vs viscosity for different propulsion types"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Speed vs Viscosity Analysis', fontsize=16, color='white')

        # Bacteria scale
        bacteria_df = df[df['scale'] == 'bacteria']
        fin_bacteria = bacteria_df[bacteria_df['propulsion'] == 'Fin']
        flag_bacteria = bacteria_df[bacteria_df['propulsion'] == 'Flagella']

        ax1.loglog(fin_bacteria['mu'], fin_bacteria['velocity'], 'o-', color='#00b4ff', label='Fins', markersize=4)
        ax1.loglog(flag_bacteria['mu'], flag_bacteria['velocity'], 's-', color='#ffb400', label='Flagella',
                   markersize=4)
        ax1.set_title('Bacteria Scale (5μm)', color='white')
        ax1.set_xlabel('Viscosity (Pa·s)', color='white')
        ax1.set_ylabel('Speed (m/s)', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Fish scale
        fish_df = df[df['scale'] == 'fish']
        fin_fish = fish_df[fish_df['propulsion'] == 'Fin']
        flag_fish = fish_df[fish_df['propulsion'] == 'Flagella']

        ax2.loglog(fin_fish['mu'], fin_fish['velocity'], 'o-', color='#00b4ff', label='Fins', markersize=4)
        ax2.loglog(flag_fish['mu'], flag_fish['velocity'], 's-', color='#ffb400', label='Flagella', markersize=4)
        ax2.set_title('Fish Scale (10cm)', color='white')
        ax2.set_xlabel('Viscosity (Pa·s)', color='white')
        ax2.set_ylabel('Speed (m/s)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Speed ratio analysis
        if len(fin_bacteria) > 0 and len(flag_bacteria) > 0:
            # Match viscosities for ratio calculation
            common_mu = np.intersect1d(fin_bacteria['mu'].round(10), flag_bacteria['mu'].round(10))
            if len(common_mu) > 10:
                fin_speeds = [fin_bacteria[fin_bacteria['mu'].round(10) == mu]['velocity'].iloc[0] for mu in
                              common_mu[:30]]
                flag_speeds = [flag_bacteria[flag_bacteria['mu'].round(10) == mu]['velocity'].iloc[0] for mu in
                               common_mu[:30]]
                ratios = np.array(fin_speeds) / np.array(flag_speeds)

                ax3.semilogx(common_mu[:30], ratios, 'o-', color='#ff6b6b', markersize=4)
                ax3.axhline(y=1, color='white', linestyle='--', alpha=0.5)
                ax3.set_title('Speed Ratio: Fins/Flagella (Bacteria)', color='white')
                ax3.set_xlabel('Viscosity (Pa·s)', color='white')
                ax3.set_ylabel('Speed Ratio', color='white')
                ax3.grid(True, alpha=0.3)

        # Reynolds number vs viscosity
        ax4.loglog(bacteria_df['mu'], bacteria_df['reynolds'], 'o', alpha=0.6, markersize=3)
        ax4.set_title('Reynolds Number vs Viscosity', color='white')
        ax4.set_xlabel('Viscosity (Pa·s)', color='white')
        ax4.set_ylabel('Reynolds Number', color='white')
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Re = 1')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/01_viscosity_vs_speed.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_reynolds_analysis(self, df):
        """Reynolds number analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Reynolds Number Analysis', fontsize=16, color='white')

        # Efficiency vs Reynolds
        fin_data = df[df['propulsion'] == 'Fin']
        flag_data = df[df['propulsion'] == 'Flagella']

        ax1.semilogx(fin_data['reynolds'], fin_data['efficiency'], 'o', alpha=0.6, color='#00b4ff', label='Fins',
                     markersize=3)
        ax1.semilogx(flag_data['reynolds'], flag_data['efficiency'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                     markersize=3)
        ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Efficiency vs Reynolds Number', color='white')
        ax1.set_xlabel('Reynolds Number', color='white')
        ax1.set_ylabel('Efficiency', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Speed vs Reynolds
        ax2.loglog(fin_data['reynolds'], fin_data['velocity'], 'o', alpha=0.6, color='#00b4ff', label='Fins',
                   markersize=3)
        ax2.loglog(flag_data['reynolds'], flag_data['velocity'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                   markersize=3)
        ax2.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Speed vs Reynolds Number', color='white')
        ax2.set_xlabel('Reynolds Number', color='white')
        ax2.set_ylabel('Speed (m/s)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Drag coefficient analysis
        drag_coeffs = []
        reynolds_nums = []
        for _, row in df.iterrows():
            if row['velocity'] > 0:
                r = row['L'] / 2
                A = math.pi * r * r
                rho = row['rho']
                v = row['velocity']
                drag = row['drag']
                cd_effective = 2 * drag / (rho * A * v * v) if v > 0 else 0
                if cd_effective > 0 and cd_effective < 1000:
                    drag_coeffs.append(cd_effective)
                    reynolds_nums.append(row['reynolds'])

        ax3.loglog(reynolds_nums, drag_coeffs, 'o', alpha=0.6, markersize=3, color='#ff6b6b')
        # Theoretical drag coefficient
        re_theory = np.logspace(-2, 6, 100)
        cd_theory = [cd_blend(re) for re in re_theory]
        ax3.loglog(re_theory, cd_theory, '-', color='white', linewidth=2, label='Theoretical')
        ax3.set_title('Drag Coefficient vs Reynolds Number', color='white')
        ax3.set_xlabel('Reynolds Number', color='white')
        ax3.set_ylabel('Drag Coefficient', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Reynolds vs Size
        bacteria_df = df[df['scale'] == 'bacteria']
        fish_df = df[df['scale'] == 'fish']
        ax4.loglog(bacteria_df['L'], bacteria_df['reynolds'], 'o', alpha=0.6, color='#ff6b6b', label='Bacteria',
                   markersize=3)
        ax4.loglog(fish_df['L'], fish_df['reynolds'], 's', alpha=0.6, color='#6bff6b', label='Fish', markersize=3)
        ax4.set_title('Reynolds Number vs Size', color='white')
        ax4.set_xlabel('Size (m)', color='white')
        ax4.set_ylabel('Reynolds Number', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/02_reynolds_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_size_vs_performance(self, df):
        """Size vs performance analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Size vs Performance Analysis', fontsize=16, color='white')

        fin_data = df[df['propulsion'] == 'Fin']
        flag_data = df[df['propulsion'] == 'Flagella']

        # Speed vs Size
        ax1.loglog(fin_data['L'], fin_data['velocity'], 'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax1.loglog(flag_data['L'], flag_data['velocity'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                   markersize=3)
        ax1.axvline(x=1e-4, color='white', linestyle='--', alpha=0.5, label='Bacteria/Fish boundary')
        ax1.set_title('Speed vs Size', color='white')
        ax1.set_xlabel('Size (m)', color='white')
        ax1.set_ylabel('Speed (m/s)', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Power vs Size
        ax2.loglog(fin_data['L'], fin_data['power'], 'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax2.loglog(flag_data['L'], flag_data['power'], 's', alpha=0.6, color='#ffb400', label='Flagella', markersize=3)
        ax2.set_title('Power vs Size', color='white')
        ax2.set_xlabel('Size (m)', color='white')
        ax2.set_ylabel('Power (W)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Efficiency vs Size
        ax3.semilogx(fin_data['L'], fin_data['efficiency'], 'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax3.semilogx(flag_data['L'], flag_data['efficiency'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                     markersize=3)
        ax3.set_title('Efficiency vs Size', color='white')
        ax3.set_xlabel('Size (m)', color='white')
        ax3.set_ylabel('Efficiency', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Speed ratio vs Size
        size_bins = np.logspace(-6, 0, 50)
        ratios = []
        sizes_used = []

        for size in size_bins:
            fin_near = fin_data[np.abs(np.log10(fin_data['L']) - np.log10(size)) < 0.1]
            flag_near = flag_data[np.abs(np.log10(flag_data['L']) - np.log10(size)) < 0.1]

            if len(fin_near) > 0 and len(flag_near) > 0:
                ratio = fin_near['velocity'].mean() / flag_near['velocity'].mean()
                ratios.append(ratio)
                sizes_used.append(size)

        ax4.semilogx(sizes_used, ratios, 'o-', color='#ff6b6b', markersize=4)
        ax4.axhline(y=1, color='white', linestyle='--', alpha=0.5)
        ax4.set_title('Speed Ratio: Fins/Flagella vs Size', color='white')
        ax4.set_xlabel('Size (m)', color='white')
        ax4.set_ylabel('Speed Ratio', color='white')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/03_size_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_frequency_analysis(self, df):
        """Frequency analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Frequency Analysis', fontsize=16, color='white')

        fin_data = df[df['propulsion'] == 'Fin']
        flag_data = df[df['propulsion'] == 'Flagella']
        bacteria_df = df[df['scale'] == 'bacteria']
        fish_df = df[df['scale'] == 'fish']

        # Speed vs Frequency (bacteria)
        bacteria_fin = bacteria_df[bacteria_df['propulsion'] == 'Fin']
        bacteria_flag = bacteria_df[bacteria_df['propulsion'] == 'Flagella']

        ax1.loglog(bacteria_fin['freq'], bacteria_fin['velocity'], 'o-', color='#00b4ff', label='Fins', markersize=4)
        ax1.loglog(bacteria_flag['freq'], bacteria_flag['velocity'], 's-', color='#ffb400', label='Flagella',
                   markersize=4)
        ax1.set_title('Speed vs Frequency (Bacteria)', color='white')
        ax1.set_xlabel('Frequency (Hz)', color='white')
        ax1.set_ylabel('Speed (m/s)', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Speed vs Frequency (fish)
        fish_fin = fish_df[fish_df['propulsion'] == 'Fin']
        fish_flag = fish_df[fish_df['propulsion'] == 'Flagella']

        ax2.loglog(fish_fin['freq'], fish_fin['velocity'], 'o-', color='#00b4ff', label='Fins', markersize=4)
        ax2.loglog(fish_flag['freq'], fish_flag['velocity'], 's-', color='#ffb400', label='Flagella', markersize=4)
        ax2.set_title('Speed vs Frequency (Fish)', color='white')
        ax2.set_xlabel('Frequency (Hz)', color='white')
        ax2.set_ylabel('Speed (m/s)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Power vs Frequency
        ax3.loglog(fin_data['freq'], fin_data['power'], 'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax3.loglog(flag_data['freq'], flag_data['power'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                   markersize=3)
        ax3.set_title('Power vs Frequency', color='white')
        ax3.set_xlabel('Frequency (Hz)', color='white')
        ax3.set_ylabel('Power (W)', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Efficiency vs Frequency
        ax4.semilogx(fin_data['freq'], fin_data['efficiency'], 'o', alpha=0.6, color='#00b4ff', label='Fins',
                     markersize=3)
        ax4.semilogx(flag_data['freq'], flag_data['efficiency'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                     markersize=3)
        ax4.set_title('Efficiency vs Frequency', color='white')
        ax4.set_xlabel('Frequency (Hz)', color='white')
        ax4.set_ylabel('Efficiency', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/04_frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_efficiency_analysis(self, df):
        """Efficiency analysis across different parameters"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Efficiency Analysis', fontsize=16, color='white')

        # Efficiency distribution
        fin_eff = df[df['propulsion'] == 'Fin']['efficiency']
        flag_eff = df[df['propulsion'] == 'Flagella']['efficiency']

        ax1.hist(fin_eff, bins=50, alpha=0.7, color='#00b4ff', label='Fins', density=True)
        ax1.hist(flag_eff, bins=50, alpha=0.7, color='#ffb400', label='Flagella', density=True)
        ax1.set_title('Efficiency Distribution', color='white')
        ax1.set_xlabel('Efficiency', color='white')
        ax1.set_ylabel('Density', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Efficiency vs Speed
        ax2.scatter(df[df['propulsion'] == 'Fin']['velocity'], df[df['propulsion'] == 'Fin']['efficiency'],
                    alpha=0.6, color='#00b4ff', label='Fins', s=20)
        ax2.scatter(df[df['propulsion'] == 'Flagella']['velocity'], df[df['propulsion'] == 'Flagella']['efficiency'],
                    alpha=0.6, color='#ffb400', label='Flagella', s=20)
        ax2.set_title('Efficiency vs Speed', color='white')
        ax2.set_xlabel('Speed (m/s)', color='white')
        ax2.set_ylabel('Efficiency', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Efficiency vs Density
        ax3.semilogx(df[df['propulsion'] == 'Fin']['rho'], df[df['propulsion'] == 'Fin']['efficiency'],
                     'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax3.semilogx(df[df['propulsion'] == 'Flagella']['rho'], df[df['propulsion'] == 'Flagella']['efficiency'],
                     's', alpha=0.6, color='#ffb400', label='Flagella', markersize=3)
        ax3.set_title('Efficiency vs Fluid Density', color='white')
        ax3.set_xlabel('Density (kg/m³)', color='white')
        ax3.set_ylabel('Efficiency', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Power efficiency (speed per unit power)
        power_eff_fin = df[df['propulsion'] == 'Fin']['velocity'] / (df[df['propulsion'] == 'Fin']['power'] + 1e-12)
        power_eff_flag = df[df['propulsion'] == 'Flagella']['velocity'] / (
                    df[df['propulsion'] == 'Flagella']['power'] + 1e-12)

        ax4.hist(power_eff_fin, bins=50, alpha=0.7, color='#00b4ff', label='Fins', density=True)
        ax4.hist(power_eff_flag, bins=50, alpha=0.7, color='#ffb400', label='Flagella', density=True)
        ax4.set_title('Power Efficiency Distribution (Speed/Power)', color='white')
        ax4.set_xlabel('Speed per Unit Power (m/s/W)', color='white')
        ax4.set_ylabel('Density', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/05_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_power_analysis(self, df):
        """Power consumption and requirements analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Power Analysis', fontsize=16, color='white')

        fin_data = df[df['propulsion'] == 'Fin']
        flag_data = df[df['propulsion'] == 'Flagella']

        # Power vs Size
        ax1.loglog(fin_data['L'], fin_data['power'], 'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax1.loglog(flag_data['L'], flag_data['power'], 's', alpha=0.6, color='#ffb400', label='Flagella', markersize=3)
        ax1.set_title('Power vs Size', color='white')
        ax1.set_xlabel('Size (m)', color='white')
        ax1.set_ylabel('Power (W)', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Power vs Speed
        ax2.loglog(fin_data['velocity'], fin_data['power'], 'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax2.loglog(flag_data['velocity'], flag_data['power'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                   markersize=3)
        ax2.set_title('Power vs Speed', color='white')
        ax2.set_xlabel('Speed (m/s)', color='white')
        ax2.set_ylabel('Power (W)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Specific power (power per unit mass)
        for _, row in df.iterrows():
            r = row['L'] / 2
            mass = row['rho'] * (4 / 3) * math.pi * r ** 3
            specific_power = row['power'] / mass if mass > 0 else 0
            df.loc[df.index == _, 'specific_power'] = specific_power

        ax3.loglog(df[df['propulsion'] == 'Fin']['L'], df[df['propulsion'] == 'Fin']['specific_power'],
                   'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax3.loglog(df[df['propulsion'] == 'Flagella']['L'], df[df['propulsion'] == 'Flagella']['specific_power'],
                   's', alpha=0.6, color='#ffb400', label='Flagella', markersize=3)
        ax3.set_title('Specific Power vs Size', color='white')
        ax3.set_xlabel('Size (m)', color='white')
        ax3.set_ylabel('Specific Power (W/kg)', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Power scaling with Reynolds number
        ax4.loglog(fin_data['reynolds'], fin_data['power'], 'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax4.loglog(flag_data['reynolds'], flag_data['power'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                   markersize=3)
        ax4.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('Power vs Reynolds Number', color='white')
        ax4.set_xlabel('Reynolds Number', color='white')
        ax4.set_ylabel('Power (W)', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/06_power_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_environment_analysis(self, df):
        """Environment-specific performance analysis"""
        # Filter for environment data
        env_df = df[df.get('environment').notna()] if 'environment' in df.columns else df

        if env_df.empty:
            print("No environment-specific data found, skipping environment analysis")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Environment-Specific Performance Analysis', fontsize=16, color='white')

        # Performance by environment
        envs = ['Water', 'Honey', 'Air', 'Oil', 'Cytoplasm']
        colors = ['#6496ff', '#ffb400', '#c8e6ff', '#8b5a2b', '#96ff96']

        # Speed comparison by environment
        env_speeds_fin = []
        env_speeds_flag = []
        env_names = []

        for env in envs:
            if env in ENVIRONMENTS:
                env_data = df[(df['rho'] == ENVIRONMENTS[env]['rho']) &
                              (df['mu'] == ENVIRONMENTS[env]['mu'])]
                if not env_data.empty:
                    fin_speed = env_data[env_data['propulsion'] == 'Fin']['velocity'].mean()
                    flag_speed = env_data[env_data['propulsion'] == 'Flagella']['velocity'].mean()
                    env_speeds_fin.append(fin_speed)
                    env_speeds_flag.append(flag_speed)
                    env_names.append(env)

        if env_names:
            x = np.arange(len(env_names))
            width = 0.35

            ax1.bar(x - width / 2, env_speeds_fin, width, label='Fins', color='#00b4ff', alpha=0.8)
            ax1.bar(x + width / 2, env_speeds_flag, width, label='Flagella', color='#ffb400', alpha=0.8)
            ax1.set_title('Average Speed by Environment', color='white')
            ax1.set_xlabel('Environment', color='white')
            ax1.set_ylabel('Speed (m/s)', color='white')
            ax1.set_xticks(x)
            ax1.set_xticklabels(env_names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Viscosity vs density map with performance overlay
        viscosities = [ENVIRONMENTS[env]['mu'] for env in envs]
        densities = [ENVIRONMENTS[env]['rho'] for env in envs]

        ax2.scatter(viscosities, densities, c=colors, s=100, alpha=0.8)
        for i, env in enumerate(envs):
            ax2.annotate(env, (viscosities[i], densities[i]),
                         xytext=(5, 5), textcoords='offset points', color='white')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_title('Environment Property Map', color='white')
        ax2.set_xlabel('Viscosity (Pa·s)', color='white')
        ax2.set_ylabel('Density (kg/m³)', color='white')
        ax2.grid(True, alpha=0.3)

        # Performance ratio by environment
        ratios = []
        for i in range(len(env_speeds_fin)):
            if env_speeds_flag[i] > 0:
                ratios.append(env_speeds_fin[i] / env_speeds_flag[i])
            else:
                ratios.append(0)

        bars = ax3.bar(env_names, ratios, color=['red' if r < 1 else 'green' for r in ratios], alpha=0.7)
        ax3.axhline(y=1, color='white', linestyle='--', alpha=0.7)
        ax3.set_title('Speed Ratio: Fins/Flagella by Environment', color='white')
        ax3.set_xlabel('Environment', color='white')
        ax3.set_ylabel('Speed Ratio', color='white')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # Environment suitability index
        suitability_fin = []
        suitability_flag = []

        for i in range(len(env_names)):
            # Higher speed and efficiency = higher suitability
            speed_fin = env_speeds_fin[i] if i < len(env_speeds_fin) else 0
            speed_flag = env_speeds_flag[i] if i < len(env_speeds_flag) else 0

            suitability_fin.append(speed_fin * 100)  # Scale for visibility
            suitability_flag.append(speed_flag * 100)

        x = np.arange(len(env_names))
        ax4.bar(x - width / 2, suitability_fin, width, label='Fins', color='#00b4ff', alpha=0.8)
        ax4.bar(x + width / 2, suitability_flag, width, label='Flagella', color='#ffb400', alpha=0.8)
        ax4.set_title('Environment Suitability Index', color='white')
        ax4.set_xlabel('Environment', color='white')
        ax4.set_ylabel('Suitability (Speed × 100)', color='white')
        ax4.set_xticks(x)
        ax4.set_xticklabels(env_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/07_environment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_thrust_drag_analysis(self, df):
        """Thrust and drag force analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Thrust and Drag Analysis', fontsize=16, color='white')

        fin_data = df[df['propulsion'] == 'Fin']
        flag_data = df[df['propulsion'] == 'Flagella']

        # Thrust vs Speed
        ax1.loglog(fin_data['velocity'], fin_data['thrust'], 'o', alpha=0.6, color='#00b4ff', label='Fins',
                   markersize=3)
        ax1.loglog(flag_data['velocity'], flag_data['thrust'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                   markersize=3)
        ax1.set_title('Thrust vs Speed', color='white')
        ax1.set_xlabel('Speed (m/s)', color='white')
        ax1.set_ylabel('Thrust (N)', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drag vs Speed
        ax2.loglog(fin_data['velocity'], fin_data['drag'], 'o', alpha=0.6, color='#00b4ff', label='Fins', markersize=3)
        ax2.loglog(flag_data['velocity'], flag_data['drag'], 's', alpha=0.6, color='#ffb400', label='Flagella',
                   markersize=3)
        ax2.set_title('Drag vs Speed', color='white')
        ax2.set_xlabel('Speed (m/s)', color='white')
        ax2.set_ylabel('Drag (N)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Thrust-to-drag ratio
        thrust_drag_ratio_fin = fin_data['thrust'] / (fin_data['drag'] + 1e-12)
        thrust_drag_ratio_flag = flag_data['thrust'] / (flag_data['drag'] + 1e-12)

        ax3.semilogx(fin_data['reynolds'], thrust_drag_ratio_fin, 'o', alpha=0.6, color='#00b4ff', label='Fins',
                     markersize=3)
        ax3.semilogx(flag_data['reynolds'], thrust_drag_ratio_flag, 's', alpha=0.6, color='#ffb400', label='Flagella',
                     markersize=3)
        ax3.axhline(y=1, color='white', linestyle='--', alpha=0.7, label='Equilibrium')
        ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Thrust/Drag Ratio vs Reynolds Number', color='white')
        ax3.set_xlabel('Reynolds Number', color='white')
        ax3.set_ylabel('Thrust/Drag Ratio', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Force scaling with size
        ax4.loglog(fin_data['L'], fin_data['thrust'], 'o', alpha=0.6, color='#00b4ff', label='Thrust (Fins)',
                   markersize=3)
        ax4.loglog(fin_data['L'], fin_data['drag'], '^', alpha=0.6, color='#00b4ff', label='Drag (Fins)', markersize=3)
        ax4.loglog(flag_data['L'], flag_data['thrust'], 's', alpha=0.6, color='#ffb400', label='Thrust (Flagella)',
                   markersize=3)
        ax4.loglog(flag_data['L'], flag_data['drag'], 'v', alpha=0.6, color='#ffb400', label='Drag (Flagella)',
                   markersize=3)
        ax4.set_title('Forces vs Size', color='white')
        ax4.set_xlabel('Size (m)', color='white')
        ax4.set_ylabel('Force (N)', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/08_thrust_drag_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_regime_maps(self, df):
        """Performance regime maps and phase diagrams"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Regime Maps', fontsize=16, color='white')

        # Reynolds vs Size regime map
        bacteria_df = df[df['scale'] == 'bacteria']
        fish_df = df[df['scale'] == 'fish']

        ax1.loglog(bacteria_df['L'], bacteria_df['reynolds'], 'o', alpha=0.6, color='#ff6b6b', label='Bacteria',
                   markersize=3)
        ax1.loglog(fish_df['L'], fish_df['reynolds'], 's', alpha=0.6, color='#6bff6b', label='Fish', markersize=3)
        ax1.axhline(y=1, color='white', linestyle='--', alpha=0.7, label='Re = 1')
        ax1.axhline(y=100, color='yellow', linestyle='--', alpha=0.7, label='Re = 100')
        ax1.set_title('Reynolds Number vs Size Regime Map', color='white')
        ax1.set_xlabel('Size (m)', color='white')
        ax1.set_ylabel('Reynolds Number', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Viscosity vs Speed performance map
        # Create a 2D histogram showing which propulsion is better
        fin_data = df[df['propulsion'] == 'Fin']
        flag_data = df[df['propulsion'] == 'Flagella']

        # Create performance difference map
        mu_bins = np.logspace(-6, 2, 20)
        v_bins = np.logspace(-8, 2, 20)
        performance_map = np.zeros((len(mu_bins) - 1, len(v_bins) - 1))

        for i in range(len(mu_bins) - 1):
            for j in range(len(v_bins) - 1):
                mu_mask = (df['mu'] >= mu_bins[i]) & (df['mu'] < mu_bins[i + 1])
                v_mask = (df['velocity'] >= v_bins[j]) & (df['velocity'] < v_bins[j + 1])
                subset = df[mu_mask & v_mask]

                if len(subset) > 0:
                    fin_perf = subset[subset['propulsion'] == 'Fin']['velocity'].mean()
                    flag_perf = subset[subset['propulsion'] == 'Flagella']['velocity'].mean()
                    if not np.isnan(fin_perf) and not np.isnan(flag_perf):
                        performance_map[i, j] = fin_perf - flag_perf

        im = ax2.imshow(performance_map, extent=[np.log10(v_bins[0]), np.log10(v_bins[-1]),
                                                 np.log10(mu_bins[0]), np.log10(mu_bins[-1])],
                        aspect='auto', origin='lower', cmap='RdBu', alpha=0.7)
        ax2.set_title('Performance Map: Fins vs Flagella', color='white')
        ax2.set_xlabel('Log₁₀(Speed) (m/s)', color='white')
        ax2.set_ylabel('Log₁₀(Viscosity) (Pa·s)', color='white')
        plt.colorbar(im, ax=ax2, label='Speed Difference (Fin - Flagella)')

        # Efficiency regime map
        ax3.scatter(df[df['propulsion'] == 'Fin']['reynolds'], df[df['propulsion'] == 'Fin']['efficiency'],
                    c=df[df['propulsion'] == 'Fin']['L'], alpha=0.6, s=20, cmap='viridis', marker='o', label='Fins')
        ax3.scatter(df[df['propulsion'] == 'Flagella']['reynolds'], df[df['propulsion'] == 'Flagella']['efficiency'],
                    c=df[df['propulsion'] == 'Flagella']['L'], alpha=0.6, s=20, cmap='plasma', marker='s',
                    label='Flagella')
        ax3.set_xscale('log')
        ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Efficiency vs Reynolds (colored by size)', color='white')
        ax3.set_xlabel('Reynolds Number', color='white')
        ax3.set_ylabel('Efficiency', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Frequency-Size optimization map
        freq_bins = np.logspace(-1, 3, 15)
        size_bins = np.logspace(-6, 0, 15)
        opt_map = np.zeros((len(freq_bins) - 1, len(size_bins) - 1))

        for i in range(len(freq_bins) - 1):
            for j in range(len(size_bins) - 1):
                freq_mask = (df['freq'] >= freq_bins[i]) & (df['freq'] < freq_bins[i + 1])
                size_mask = (df['L'] >= size_bins[j]) & (df['L'] < size_bins[j + 1])
                subset = df[freq_mask & size_mask]

                if len(subset) > 0:
                    fin_eff = subset[subset['propulsion'] == 'Fin']['efficiency'].mean()
                    flag_eff = subset[subset['propulsion'] == 'Flagella']['efficiency'].mean()
                    if not np.isnan(fin_eff) and not np.isnan(flag_eff):
                        opt_map[i, j] = 1 if fin_eff > flag_eff else -1

        im2 = ax4.imshow(opt_map, extent=[np.log10(size_bins[0]), np.log10(size_bins[-1]),
                                          np.log10(freq_bins[0]), np.log10(freq_bins[-1])],
                         aspect='auto', origin='lower', cmap='RdYlBu', alpha=0.8)
        ax4.set_title('Optimal Propulsion Map (Freq vs Size)', color='white')
        ax4.set_xlabel('Log₁₀(Size) (m)', color='white')
        ax4.set_ylabel('Log₁₀(Frequency) (Hz)', color='white')
        plt.colorbar(im2, ax=ax4, label='Optimal (Blue=Fins, Red=Flagella)')

        plt.tight_layout()
        plt.savefig('analysis_graphs/09_performance_regime_maps.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_scaling_laws(self, df):
        """Scaling laws and allometric relationships"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scaling Laws and Allometric Relationships', fontsize=16, color='white')

        # Speed vs Size scaling
        fin_data = df[df['propulsion'] == 'Fin']
        flag_data = df[df['propulsion'] == 'Flagella']

        # Fit power laws
        fin_sizes = fin_data['L'].values
        fin_speeds = fin_data['velocity'].values
        flag_sizes = flag_data['L'].values
        flag_speeds = flag_data['velocity'].values

        # Remove zeros and invalid values
        fin_valid = (fin_sizes > 0) & (fin_speeds > 0)
        flag_valid = (flag_sizes > 0) & (flag_speeds > 0)

        if np.sum(fin_valid) > 10 and np.sum(flag_valid) > 10:
            # Fit power law: speed = a * size^b
            fin_log_fit = np.polyfit(np.log(fin_sizes[fin_valid]), np.log(fin_speeds[fin_valid]), 1)
            flag_log_fit = np.polyfit(np.log(flag_sizes[flag_valid]), np.log(flag_speeds[flag_valid]), 1)

            size_range = np.logspace(-6, 0, 100)
            fin_fit_line = np.exp(fin_log_fit[1]) * size_range ** fin_log_fit[0]
            flag_fit_line = np.exp(flag_log_fit[1]) * size_range ** flag_log_fit[0]

            ax1.loglog(fin_sizes[fin_valid], fin_speeds[fin_valid], 'o', alpha=0.6, color='#00b4ff', markersize=3)
            ax1.loglog(flag_sizes[flag_valid], flag_speeds[flag_valid], 's', alpha=0.6, color='#ffb400', markersize=3)
            ax1.loglog(size_range, fin_fit_line, '-', color='#00b4ff',
                       label=f'Fins: V ∝ L^{fin_log_fit[0]:.2f}', linewidth=2)
            ax1.loglog(size_range, flag_fit_line, '-', color='#ffb400',
                       label=f'Flagella: V ∝ L^{flag_log_fit[0]:.2f}', linewidth=2)

        ax1.set_title('Speed vs Size Scaling Law', color='white')
        ax1.set_xlabel('Size (m)', color='white')
        ax1.set_ylabel('Speed (m/s)', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Power vs Size scaling
        fin_powers = fin_data['power'].values
        flag_powers = flag_data['power'].values

        fin_power_valid = (fin_sizes > 0) & (fin_powers > 0)
        flag_power_valid = (flag_sizes > 0) & (flag_powers > 0)

        if np.sum(fin_power_valid) > 10 and np.sum(flag_power_valid) > 10:
            fin_power_fit = np.polyfit(np.log(fin_sizes[fin_power_valid]), np.log(fin_powers[fin_power_valid]), 1)
            flag_power_fit = np.polyfit(np.log(flag_sizes[flag_power_valid]), np.log(flag_powers[flag_power_valid]), 1)

            fin_power_line = np.exp(fin_power_fit[1]) * size_range ** fin_power_fit[0]
            flag_power_line = np.exp(flag_power_fit[1]) * size_range ** flag_power_fit[0]

            ax2.loglog(fin_sizes[fin_power_valid], fin_powers[fin_power_valid], 'o', alpha=0.6, color='#00b4ff',
                       markersize=3)
            ax2.loglog(flag_sizes[flag_power_valid], flag_powers[flag_power_valid], 's', alpha=0.6, color='#ffb400',
                       markersize=3)
            ax2.loglog(size_range, fin_power_line, '-', color='#00b4ff',
                       label=f'Fins: P ∝ L^{fin_power_fit[0]:.2f}', linewidth=2)
            ax2.loglog(size_range, flag_power_line, '-', color='#ffb400',
                       label=f'Flagella: P ∝ L^{flag_power_fit[0]:.2f}', linewidth=2)

        ax2.set_title('Power vs Size Scaling Law', color='white')
        ax2.set_xlabel('Size (m)', color='white')
        ax2.set_ylabel('Power (W)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Reynolds scaling with various parameters
        re_data = df['reynolds'].values
        freq_data = df['freq'].values
        mu_data = df['mu'].values

        valid_re = (re_data > 0) & (freq_data > 0)
        if np.sum(valid_re) > 10:
            ax3.loglog(freq_data[valid_re], re_data[valid_re], 'o', alpha=0.4, markersize=3, color='#ff6b6b')

        ax3.set_title('Reynolds Number vs Frequency', color='white')
        ax3.set_xlabel('Frequency (Hz)', color='white')
        ax3.set_ylabel('Reynolds Number', color='white')
        ax3.grid(True, alpha=0.3)

        # Efficiency scaling
        eff_data = df['efficiency'].values
        valid_eff = (re_data > 0) & (eff_data > 0)

        if np.sum(valid_eff) > 10:
            ax4.semilogx(re_data[valid_eff], eff_data[valid_eff], 'o', alpha=0.4, markersize=3, color='#6bff6b')

            # Add theoretical efficiency curves
            re_theory = np.logspace(-6, 6, 100)
            fin_eff_theory = [fin_eff(re) for re in re_theory]
            flag_eff_theory = [flag_eff(re) for re in re_theory]

            ax4.semilogx(re_theory, fin_eff_theory, '-', color='#00b4ff', linewidth=2, label='Fin Theory')
            ax4.semilogx(re_theory, flag_eff_theory, '-', color='#ffb400', linewidth=2, label='Flagella Theory')

        ax4.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('Efficiency vs Reynolds Number', color='white')
        ax4.set_xlabel('Reynolds Number', color='white')
        ax4.set_ylabel('Efficiency', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/10_scaling_laws.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_optimal_frequency(self, df):
        """Optimal frequency analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Optimal Frequency Analysis', fontsize=16, color='white')

        # Speed vs Frequency for different sizes
        sizes_to_analyze = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(sizes_to_analyze)))

        for i, size in enumerate(sizes_to_analyze):
            size_data = df[np.abs(np.log10(df['L']) - np.log10(size)) < 0.2]
            if not size_data.empty:
                fin_data = size_data[size_data['propulsion'] == 'Fin']
                if not fin_data.empty:
                    sorted_data = fin_data.sort_values('freq')
                    ax1.semilogx(sorted_data['freq'], sorted_data['velocity'],
                                 'o-', color=colors[i], alpha=0.7, markersize=3,
                                 label=f'L={size:.0e}m')

        ax1.set_title('Speed vs Frequency - Fins (Different Sizes)', color='white')
        ax1.set_xlabel('Frequency (Hz)', color='white')
        ax1.set_ylabel('Speed (m/s)', color='white')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Same for flagella
        for i, size in enumerate(sizes_to_analyze):
            size_data = df[np.abs(np.log10(df['L']) - np.log10(size)) < 0.2]
            if not size_data.empty:
                flag_data = size_data[size_data['propulsion'] == 'Flagella']
                if not flag_data.empty:
                    sorted_data = flag_data.sort_values('freq')
                    ax2.semilogx(sorted_data['freq'], sorted_data['velocity'],
                                 's-', color=colors[i], alpha=0.7, markersize=3,
                                 label=f'L={size:.0e}m')

        ax2.set_title('Speed vs Frequency - Flagella (Different Sizes)', color='white')
        ax2.set_xlabel('Frequency (Hz)', color='white')
        ax2.set_ylabel('Speed (m/s)', color='white')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Find optimal frequencies
        optimal_freqs_fin = []
        optimal_freqs_flag = []
        optimal_sizes = []

        for size in sizes_to_analyze:
            size_data = df[np.abs(np.log10(df['L']) - np.log10(size)) < 0.2]
            if not size_data.empty:
                fin_data = size_data[size_data['propulsion'] == 'Fin']
                flag_data = size_data[size_data['propulsion'] == 'Flagella']

                if not fin_data.empty:
                    best_fin = fin_data.loc[fin_data['velocity'].idxmax()]
                    optimal_freqs_fin.append(best_fin['freq'])
                    optimal_sizes.append(size)

                if not flag_data.empty:
                    best_flag = flag_data.loc[flag_data['velocity'].idxmax()]
                    optimal_freqs_flag.append(best_flag['freq'])

        if optimal_freqs_fin and optimal_freqs_flag:
            ax3.loglog(optimal_sizes, optimal_freqs_fin, 'o-', color='#00b4ff',
                       label='Fins', markersize=6, linewidth=2)
            ax3.loglog(optimal_sizes, optimal_freqs_flag, 's-', color='#ffb400',
                       label='Flagella', markersize=6, linewidth=2)

        ax3.set_title('Optimal Frequency vs Size', color='white')
        ax3.set_xlabel('Size (m)', color='white')
        ax3.set_ylabel('Optimal Frequency (Hz)', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Power vs Frequency optimization
        bacteria_data = df[df['scale'] == 'bacteria']
        fish_data = df[df['scale'] == 'fish']

        # Calculate power efficiency (speed/power)
        bacteria_data_copy = bacteria_data.copy()
        bacteria_data_copy['power_eff'] = bacteria_data_copy['velocity'] / (bacteria_data_copy['power'] + 1e-12)

        bacteria_fin = bacteria_data_copy[bacteria_data_copy['propulsion'] == 'Fin']
        bacteria_flag = bacteria_data_copy[bacteria_data_copy['propulsion'] == 'Flagella']

        if not bacteria_fin.empty and not bacteria_flag.empty:
            ax4.semilogx(bacteria_fin['freq'], bacteria_fin['power_eff'],
                         'o', alpha=0.6, color='#00b4ff', label='Fins (Bacteria)', markersize=3)
            ax4.semilogx(bacteria_flag['freq'], bacteria_flag['power_eff'],
                         's', alpha=0.6, color='#ffb400', label='Flagella (Bacteria)', markersize=3)

        ax4.set_title('Power Efficiency vs Frequency', color='white')
        ax4.set_xlabel('Frequency (Hz)', color='white')
        ax4.set_ylabel('Power Efficiency (m/s/W)', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_graphs/11_optimal_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_multi_parameter_analysis(self, df):
        """Multi-parameter analysis and correlations"""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Multi-Parameter Analysis and Correlations', fontsize=20, color='white')

        # Create a grid of subplots
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Correlation matrix
        ax1 = fig.add_subplot(gs[0, 0])

        # Select numerical columns for correlation
        numerical_cols = ['L', 'freq', 'rho', 'mu', 'velocity', 'reynolds', 'efficiency', 'thrust', 'drag', 'power']
        corr_data = df[numerical_cols].corr()

        im = ax1.imshow(corr_data, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(numerical_cols)))
        ax1.set_yticks(range(len(numerical_cols)))
        ax1.set_xticklabels(numerical_cols, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(numerical_cols, fontsize=8)
        ax1.set_title('Parameter Correlations', color='white', fontsize=12)
        plt.colorbar(im, ax=ax1, shrink=0.8)

        # 2. 3D performance space (Size, Frequency, Speed)
        ax2 = fig.add_subplot(gs[0, 1:3], projection='3d')

        fin_data = df[df['propulsion'] == 'Fin']
        flag_data = df[df['propulsion'] == 'Flagella']

        ax2.scatter(np.log10(fin_data['L']), np.log10(fin_data['freq']), np.log10(fin_data['velocity']),
                    c='#00b4ff', alpha=0.6, s=20, label='Fins')
        ax2.scatter(np.log10(flag_data['L']), np.log10(flag_data['freq']), np.log10(flag_data['velocity']),
                    c='#ffb400', alpha=0.6, s=20, label='Flagella')

        ax2.set_xlabel('Log₁₀(Size)', color='white')
        ax2.set_ylabel('Log₁₀(Frequency)', color='white')
        ax2.set_zlabel('Log₁₀(Speed)', color='white')
        ax2.set_title('3D Performance Space', color='white', fontsize=12)
        ax2.legend()

        # 3. Reynolds regime analysis
        ax3 = fig.add_subplot(gs[0, 3])

        low_re = df[df['reynolds'] < 1]
        mid_re = df[(df['reynolds'] >= 1) & (df['reynolds'] < 100)]
        high_re = df[df['reynolds'] >= 100]

        regimes = ['Low Re (<1)', 'Mid Re (1-100)', 'High Re (>100)']
        fin_speeds = [
            low_re[low_re['propulsion'] == 'Fin']['velocity'].mean(),
            mid_re[mid_re['propulsion'] == 'Fin']['velocity'].mean(),
            high_re[high_re['propulsion'] == 'Fin']['velocity'].mean()
        ]
        flag_speeds = [
            low_re[low_re['propulsion'] == 'Flagella']['velocity'].mean(),
            mid_re[mid_re['propulsion'] == 'Flagella']['velocity'].mean(),
            high_re[high_re['propulsion'] == 'Flagella']['velocity'].mean()
        ]

        x = np.arange(len(regimes))
        width = 0.35

        ax3.bar(x - width / 2, fin_speeds, width, label='Fins', color='#00b4ff', alpha=0.8)
        ax3.bar(x + width / 2, flag_speeds, width, label='Flagella', color='#ffb400', alpha=0.8)
        ax3.set_title('Performance by Reynolds Regime', color='white', fontsize=12)
        ax3.set_xlabel('Reynolds Regime', color='white')
        ax3.set_ylabel('Average Speed (m/s)', color='white')
        ax3.set_xticks(x)
        ax3.set_xticklabels(regimes, fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Efficiency distribution by scale
        ax4 = fig.add_subplot(gs[1, 0])

        bacteria_eff = df[df['scale'] == 'bacteria']['efficiency']
        fish_eff = df[df['scale'] == 'fish']['efficiency']

        ax4.hist(bacteria_eff, bins=30, alpha=0.7, color='#ff6b6b', label='Bacteria', density=True)
        ax4.hist(fish_eff, bins=30, alpha=0.7, color='#6bff6b', label='Fish', density=True)
        ax4.set_title('Efficiency by Scale', color='white', fontsize=12)
        ax4.set_xlabel('Efficiency', color='white')
        ax4.set_ylabel('Density', color='white')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Performance vs fluid properties heatmap
        ax5 = fig.add_subplot(gs[1, 1])

        # Create performance matrix
        rho_vals = sorted(df['rho'].unique())
        mu_vals = sorted(df['mu'].unique())

        if len(rho_vals) > 1 and len(mu_vals) > 1:
            perf_matrix = np.zeros((len(mu_vals), len(rho_vals)))

            for i, mu in enumerate(mu_vals):
                for j, rho in enumerate(rho_vals):
                    subset = df[(df['rho'] == rho) & (df['mu'] == mu)]
                    if not subset.empty:
                        perf_matrix[i, j] = subset['velocity'].mean()

            im = ax5.imshow(perf_matrix, aspect='auto', cmap='viridis', origin='lower')
            ax5.set_title('Speed vs Fluid Properties', color='white', fontsize=12)
            ax5.set_xlabel('Density Index', color='white')
            ax5.set_ylabel('Viscosity Index', color='white')
            plt.colorbar(im, ax=ax5, shrink=0.8)

        # 6. Power scaling comparison
        ax6 = fig.add_subplot(gs[1, 2])

        sizes = np.logspace(-6, 0, 50)
        theoretical_power_fin = sizes ** 3  # Volume scaling
        theoretical_power_flag = sizes ** 2  # Area scaling

        actual_sizes = df['L'].values
        actual_powers = df['power'].values

        ax6.loglog(actual_sizes[df['propulsion'] == 'Fin'], actual_powers[df['propulsion'] == 'Fin'],
                   'o', alpha=0.4, color='#00b4ff', markersize=3, label='Fins (Actual)')
        ax6.loglog(actual_sizes[df['propulsion'] == 'Flagella'], actual_powers[df['propulsion'] == 'Flagella'],
                   's', alpha=0.4, color='#ffb400', markersize=3, label='Flagella (Actual)')
        ax6.loglog(sizes, theoretical_power_fin * 1e-10, '--', color='#00b4ff',
                   linewidth=2, label='Fins (L³ scaling)')
        ax6.loglog(sizes, theoretical_power_flag * 1e-8, '--', color='#ffb400',
                   linewidth=2, label='Flagella (L² scaling)')

        ax6.set_title('Power Scaling Laws', color='white', fontsize=12)
        ax6.set_xlabel('Size (m)', color='white')
        ax6.set_ylabel('Power (W)', color='white')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        # 7. Speed ratio analysis
        ax7 = fig.add_subplot(gs[1, 3])

        # Calculate speed ratios for different parameter ranges
        viscosity_ranges = [(0, 1e-3), (1e-3, 1e-1), (1e-1, 10)]
        range_labels = ['Low μ', 'Med μ', 'High μ']
        ratios = []

        for mu_min, mu_max in viscosity_ranges:
            subset = df[(df['mu'] >= mu_min) & (df['mu'] < mu_max)]
            if not subset.empty:
                fin_speed = subset[subset['propulsion'] == 'Fin']['velocity'].mean()
                flag_speed = subset[subset['propulsion'] == 'Flagella']['velocity'].mean()
                ratio = fin_speed / flag_speed if flag_speed > 0 else 0
                ratios.append(ratio)
            else:
                ratios.append(0)

        bars = ax7.bar(range_labels, ratios, color=['green' if r > 1 else 'red' for r in ratios], alpha=0.7)
        ax7.axhline(y=1, color='white', linestyle='--', alpha=0.7)
        ax7.set_title('Speed Ratio by Viscosity', color='white', fontsize=12)
        ax7.set_xlabel('Viscosity Range', color='white')
        ax7.set_ylabel('Speed Ratio (Fin/Flag)', color='white')
        ax7.grid(True, alpha=0.3)

        # 8. Time series simulation results
        ax8 = fig.add_subplot(gs[2, :2])

        # Simulate performance over time for typical creatures
        time_steps = 100
        times = np.linspace(0, 10, time_steps)

        # Bacteria in water
        bacteria_speeds_fin = []
        bacteria_speeds_flag = []

        for t in times:
            # Add some realistic time variation
            freq_variation = 1 + 0.1 * np.sin(t * 2)

            bacteria_fin_perf = self.calculate_performance(5e-6, 100 * freq_variation, 1000, 0.001, Propulsion.FIN)
            bacteria_flag_perf = self.calculate_performance(5e-6, 100 * freq_variation, 1000, 0.001,
                                                            Propulsion.FLAGELLA)

            bacteria_speeds_fin.append(bacteria_fin_perf['velocity'])
            bacteria_speeds_flag.append(bacteria_flag_perf['velocity'])

        ax8.plot(times, bacteria_speeds_fin, '-', color='#00b4ff', linewidth=2, label='Fins')
        ax8.plot(times, bacteria_speeds_flag, '-', color='#ffb400', linewidth=2, label='Flagella')
        ax8.set_title('Time Series: Bacteria Performance (Variable Frequency)', color='white', fontsize=12)
        ax8.set_xlabel('Time (s)', color='white')
        ax8.set_ylabel('Speed (m/s)', color='white')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. Performance envelope
        ax9 = fig.add_subplot(gs[2, 2:])

        # Create performance envelope showing optimal operating conditions
        freq_range = np.logspace(0, 3, 30)
        size_range = np.logspace(-6, -4, 30)

        max_speeds_fin = np.zeros((len(freq_range), len(size_range)))
        max_speeds_flag = np.zeros((len(freq_range), len(size_range)))

        for i, freq in enumerate(freq_range):
            for j, size in enumerate(size_range):
                fin_perf = self.calculate_performance(size, freq, 1000, 0.001, Propulsion.FIN, steps=200)
                flag_perf = self.calculate_performance(size, freq, 1000, 0.001, Propulsion.FLAGELLA, steps=200)

                max_speeds_fin[i, j] = fin_perf['velocity']
                max_speeds_flag[i, j] = flag_perf['velocity']

        # Show difference
        speed_diff = max_speeds_fin - max_speeds_flag

        im = ax9.imshow(speed_diff, extent=[np.log10(size_range[0]), np.log10(size_range[-1]),
                                            np.log10(freq_range[0]), np.log10(freq_range[-1])],
                        aspect='auto', origin='lower', cmap='RdBu')
        ax9.set_title('Performance Envelope: Fins vs Flagella (Bacteria Scale)', color='white', fontsize=12)
        ax9.set_xlabel('Log₁₀(Size) (m)', color='white')
        ax9.set_ylabel('Log₁₀(Frequency) (Hz)', color='white')
        plt.colorbar(im, ax=ax9, shrink=0.8, label='Speed Difference (m/s)')

        # 10. Statistical summary
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis('off')

        # Create summary statistics text
        fin_stats = df[df['propulsion'] == 'Fin']
        flag_stats = df[df['propulsion'] == 'Flagella']

        summary_text = f"""
        STATISTICAL SUMMARY:

        FINS:                                    FLAGELLA:
        • Mean Speed: {fin_stats['velocity'].mean():.2e} m/s     • Mean Speed: {flag_stats['velocity'].mean():.2e} m/s
        • Max Speed: {fin_stats['velocity'].max():.2e} m/s       • Max Speed: {flag_stats['velocity'].max():.2e} m/s  
        • Mean Efficiency: {fin_stats['efficiency'].mean():.3f}   • Mean Efficiency: {flag_stats['efficiency'].mean():.3f}
        • Mean Power: {fin_stats['power'].mean():.2e} W          • Mean Power: {flag_stats['power'].mean():.2e} W
        • Optimal Re Range: {fin_stats[fin_stats['velocity'] == fin_stats['velocity'].max()]['reynolds'].iloc[0]:.1f}  • Optimal Re Range: {flag_stats[flag_stats['velocity'] == flag_stats['velocity'].max()]['reynolds'].iloc[0]:.1f}

        PERFORMANCE COMPARISON:
        • Fins perform better in {len(df[(df['propulsion'] == 'Fin') & (df['velocity'] > df.groupby(['L', 'freq', 'rho', 'mu'])['velocity'].transform('max') * 0.9)])} / {len(fin_stats)} conditions
        • Flagella perform better in {len(df[(df['propulsion'] == 'Flagella') & (df['velocity'] > df.groupby(['L', 'freq', 'rho', 'mu'])['velocity'].transform('max') * 0.9)])} / {len(flag_stats)} conditions
        • Speed ratio (Fin/Flagella): {(fin_stats['velocity'].mean() / flag_stats['velocity'].mean()):.2f}
        • Power ratio (Fin/Flagella): {(fin_stats['power'].mean() / flag_stats['power'].mean()):.2f}
        """

        ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=11,
                  color='white', verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig('analysis_graphs/12_multi_parameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_data_to_csv(self):
        """Save all generated data to CSV for external analysis"""
        if not self.data_points:
            print("No data to save. Generate data first.")
            return

        import pandas as pd
        df = pd.DataFrame(self.data_points)
        df.to_csv('analysis_graphs/locomotion_data.csv', index=False)
        print(f"Data saved to 'analysis_graphs/locomotion_data.csv' ({len(df)} rows)")

        # Also save summary statistics
        summary_stats = {
            'total_simulations': len(df),
            'fin_simulations': len(df[df['propulsion'] == 'Fin']),
            'flagella_simulations': len(df[df['propulsion'] == 'Flagella']),
            'bacteria_scale': len(df[df['scale'] == 'bacteria']),
            'fish_scale': len(df[df['scale'] == 'fish']),
            'parameter_ranges': {
                'size': {'min': df['L'].min(), 'max': df['L'].max()},
                'frequency': {'min': df['freq'].min(), 'max': df['freq'].max()},
                'viscosity': {'min': df['mu'].min(), 'max': df['mu'].max()},
                'density': {'min': df['rho'].min(), 'max': df['rho'].max()},
                'reynolds': {'min': df['reynolds'].min(), 'max': df['reynolds'].max()}
            }
        }

        with open('analysis_graphs/analysis_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)


# --------------------
# Enhanced Game with Analysis Integration
# --------------------
class Creature:
    def __init__(self, label: str, x: float, y: float, color: Tuple[int, int, int],
                 L: float, freq: float, propulsion: Propulsion, is_bacteria: bool = False):
        self.label = label
        self.start_x = x
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0.0, 0.0)
        self.color = color
        self.L = L
        self.freq = freq
        self.amp = max(1e-6, 0.2 * L)
        self.pitch = max(1e-6, 0.2 * L)
        self.propulsion = propulsion
        self.anim_t = 0.0
        self.Re = 0.0
        self.speed = 0.0
        self.eff = 0.0
        self.is_bacteria = is_bacteria

    def radius_px(self) -> int:
        if self.L < 1e-4: return 15
        if self.L < 1e-2: return 22
        if self.L < 1e-1: return 35
        if self.L < 5e-1: return 48
        return 60

    def update(self, dt: float, rho: float, mu: float):
        U = self.vel.x
        Re = reynolds(rho, abs(U), self.L, mu)

        if self.propulsion == Propulsion.FIN:
            eff = fin_eff(Re)
            thrust = 0.5 * rho * (self.freq * self.amp) ** 2 * self.L ** 2 * eff
        else:
            eff = flag_eff(Re)
            thrust = 6 * math.pi * mu * (self.L / 2) * (self.freq * self.pitch) * eff

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

        visual_scale = Config.SPEED_VISUAL_BACTERIA if self.is_bacteria else Config.SPEED_VISUAL_FISH
        self.pos.x += self.vel.x * dt * visual_scale

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
        pygame.draw.circle(screen, (255, 255, 255), pos, rpx, 2)

        if self.propulsion == Propulsion.FIN:
            fin_length = 2.5 * rpx
            fin_angle = math.sin(self.anim_t * 8) * 0.4
            tail_points = [
                (pos[0] - rpx, pos[1]),
                (pos[0] - rpx - fin_length, pos[1] - int(fin_length * 0.5 * math.sin(fin_angle))),
                (pos[0] - rpx - fin_length, pos[1] + int(fin_length * 0.5 * math.sin(fin_angle)))
            ]
            pygame.draw.polygon(screen, self.color, tail_points)
            pygame.draw.polygon(screen, (255, 255, 255), tail_points, 2)
        else:
            flag_points = []
            length = 3.5 * rpx
            segments = 20
            for i in range(segments + 1):
                t = i / segments
                x = pos[0] - rpx - int(t * length)
                y = pos[1] + int(0.4 * rpx * math.sin(12 * t + self.anim_t * 15))
                flag_points.append((x, y))
            if len(flag_points) > 1:
                pygame.draw.lines(screen, self.color, False, flag_points, 3)

        if show_labels:
            lbl = f"{self.label} — {self.propulsion.value}"
            font = pygame.font.SysFont("Arial", 16, bold=True)
            surf = font.render(lbl, True, self.color)
            screen.blit(surf, (pos[0] - surf.get_width() // 2, pos[1] - rpx - 25))


# --------------------
# UI
# --------------------
class UI:
    def __init__(self, font_main, font_small, font_bold):
        self.font = font_main
        self.small = font_small
        self.bold = font_bold

    def draw_grid(self, screen):
        step = 50
        for x in range(0, Config.WIDTH, step):
            pygame.draw.line(screen, Config.GRID, (x, 0), (x, Config.HEIGHT), 1)
        for y in range(0, Config.HEIGHT, step):
            pygame.draw.line(screen, Config.GRID, (0, y), (Config.WIDTH, y), 1)

    def panel(self, screen, lines: List[str], x: int, y: int, bg_color=(10, 15, 25), font=None):
        font_to_use = font if font is not None else self.font
        pads = 12
        spacing = 3
        line_surfs = [font_to_use.render(line, True, Config.TEXT) for line in lines if line.strip()]
        if not line_surfs: return
        w = max(s.get_width() for s in line_surfs) + pads * 2
        h = sum(s.get_height() for s in line_surfs) + spacing * (len(line_surfs) - 1) + pads * 2
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        fill_color = (bg_color[0], bg_color[1], bg_color[2], 210) if len(bg_color) >= 3 else (10, 15, 25, 210)
        surf.fill(fill_color)
        pygame.draw.rect(surf, (255, 255, 255, 30), surf.get_rect(), 1)
        screen.blit(surf, (x, y))
        cy = y + pads
        for s in line_surfs:
            screen.blit(s, (x + pads, cy))
            cy += s.get_height() + spacing

    def help_overlay(self, screen):
        text = [
            "LOCOMOTION DEMO - VERTICAL LAYOUT",
            "",
            "Scenarios: 1=Bacteria  2=Fish",
            "",
            "Controls:",
            "Q/A: Frequency       W/S: Size",
            "E/D: Speed Decimals  R/T: Reynolds Decimals",
            "",
            "Environments:",
            "Z=Water  X=Honey  C=Air  V=Oil  B=Cytoplasm",
            "",
            "Analysis:",
            "G: Generate comprehensive analysis graphs",
            "J: Save current data to CSV",
            "",
            "Other: F=toggle labels  H=close help"
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

    def draw_static_attributes(self, screen, creature: Creature, x: int, y: int, speed_prec: int, re_prec: int):
        lines = [
            f"{creature.propulsion.value.upper()}",
            f"Size: {creature.L:.2e} m",
            f"Freq: {creature.freq:.1f} Hz",
            f"Re ({re_prec}f): {creature.Re:.{re_prec}f}",
            f"Speed ({speed_prec}f): {creature.speed:.{speed_prec}f} m/s",
            f"Efficiency: {creature.eff:.2f}"
        ]
        bg_color = creature.color[:3] if len(creature.color) >= 3 else (10, 15, 25)
        self.panel(screen, lines, x, y, bg_color, font=self.bold)


# --------------------
# Enhanced Game
# --------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
        pygame.display.set_caption("Locomotion Demo with Analysis")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", Config.FONT_SIZE, bold=True)
        self.small = pygame.font.SysFont("Arial", 16)
        self.bold = pygame.font.SysFont("Arial", 18, bold=True)
        self.ui = UI(self.small, self.small, self.bold)

        self.current_env = "Water"
        self.rho = ENVIRONMENTS[self.current_env]["rho"]
        self.mu = ENVIRONMENTS[self.current_env]["mu"]

        self.speed_precision = 10
        self.re_precision = 10

        self.scenario = "bacteria"
        self.show_labels = True
        self.show_help = False

        # Initialize data analyzer
        self.analyzer = DataAnalyzer()
        self.analysis_status = ""
        self.analysis_timer = 0
        self.analysis_pending = False  # Add this flag

        self._build_scenario()

    def _build_scenario(self):
        cx = Config.WIDTH * 0.5
        cy_top = Config.HEIGHT * 0.3
        cy_bottom = Config.HEIGHT * 0.7
        if self.scenario == "bacteria":
            L = 5e-6;
            f = 100.0
            self.top = Creature("Bacterium", cx, cy_top, Config.FIN, L, f, Propulsion.FIN, True)
            self.bottom = Creature("Bacterium", cx, cy_bottom, Config.FLAG, L, f, Propulsion.FLAGELLA, True)
        else:
            L = 0.1;
            f = 4.0
            self.top = Creature("Fish", cx, cy_top, Config.FIN, L, f, Propulsion.FIN, False)
            self.bottom = Creature("Fish", cx, cy_bottom, Config.FLAG, L, f, Propulsion.FLAGELLA, False)

    def _generate_analysis(self):
        """Generate comprehensive analysis - runs on main thread"""
        self.analysis_status = "Generating comprehensive analysis... This may take a few minutes."
        print("Starting comprehensive analysis generation...")

        try:
            # Temporarily pause the game loop display updates
            pygame.display.set_caption("Locomotion Demo - Generating Analysis...")

            self.analyzer.create_all_graphs()
            self.analyzer.save_data_to_csv()
            self.analysis_status = "Analysis complete! Check 'analysis_graphs' folder for results."
            print("Analysis generation completed successfully!")

            pygame.display.set_caption("Locomotion Demo with Analysis")
        except Exception as e:
            self.analysis_status = f"Analysis failed: {str(e)}"
            print(f"Analysis generation failed: {e}")

        self.analysis_timer = pygame.time.get_ticks()
        self.analysis_pending = False

    def run(self):
        while True:
            dt = self.clock.tick(Config.FPS) / 1000.0
            current_time = pygame.time.get_ticks()

            if self.analysis_pending:
                self._generate_analysis()
                continue

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if e.key == pygame.K_1:
                        self.scenario = "bacteria"
                        self._build_scenario()
                    if e.key == pygame.K_2:
                        self.scenario = "fish"
                        self._build_scenario()
                    if e.key == pygame.K_f:
                        self.show_labels = not self.show_labels
                    if e.key == pygame.K_h:
                        self.show_help = not self.show_help
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
                    if e.key == pygame.K_g:
                        # Generate comprehensive analysis
                        self.analysis_pending = True
                    if e.key == pygame.K_j:
                        # Save current data
                        self._save_current_data()
                    # Precision controls
                    if e.key == pygame.K_e:
                        self.speed_precision = min(20, self.speed_precision + 1)
                    if e.key == pygame.K_d:
                        self.speed_precision = max(0, self.speed_precision - 1)
                    if e.key == pygame.K_r:
                        self.re_precision = min(20, self.re_precision + 1)
                    if e.key == pygame.K_t:
                        self.re_precision = max(0, self.re_precision - 1)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                self.top.freq = self.bottom.freq = min(1000, self.top.freq * 1.02)
            if keys[pygame.K_a]:
                self.top.freq = self.bottom.freq = max(0.1, self.top.freq * 0.98)
            if keys[pygame.K_w]:
                for c in [self.top, self.bottom]:
                    c.L = min(2.0, c.L * 1.02)
                    c.amp = max(1e-6, 0.2 * c.L)
                    c.pitch = max(1e-6, 0.2 * c.L)
            if keys[pygame.K_s]:
                for c in [self.top, self.bottom]:
                    c.L = max(1e-6, c.L * 0.98)
                    c.amp = max(1e-6, 0.2 * c.L)
                    c.pitch = max(1e-6, 0.2 * c.L)

            self.top.update(dt, self.rho, self.mu)
            self.bottom.update(dt, self.rho, self.mu)

            env_color = ENVIRONMENTS[self.current_env]["color"]
            bg_tinted = tuple(int(Config.BG[i] * 0.8 + env_color[i] * 0.2) for i in range(3))
            self.screen.fill(bg_tinted)

            self.ui.draw_grid(self.screen)
            self.top.draw(self.screen, self.show_labels)
            self.bottom.draw(self.screen, self.show_labels)

            self.ui.draw_static_attributes(self.screen, self.top, 20, int(self.top.pos.y) - 60,
                                           self.speed_precision, self.re_precision)
            self.ui.draw_static_attributes(self.screen, self.bottom, 20, int(self.bottom.pos.y) - 60,
                                           self.speed_precision, self.re_precision)

            env_lines = [
                f"Environment: {self.current_env}",
                f"Density: {self.rho:.2g} kg/m³",
                f"Viscosity: {self.mu:.2g} Pa·s",
                "",
                f"Scenario: {'Bacteria' if self.scenario == 'bacteria' else 'Fish'}",
                "",
                "Expected Winner:",
                f"• Bacteria: {'Flagella' if self.scenario == 'bacteria' else 'N/A'}",
                f"• Fish: {'Fins' if self.scenario == 'fish' else 'N/A'}"
            ]
            self.ui.panel(self.screen, env_lines, Config.WIDTH - 320, 20)

            control_lines = [
                "Controls:",
                "Q/A=freq  W/S=size",
                "E/D=spd prec, R/T=Re prec",
                "Env: Z/X/C/V/B",
                "1/2=scenario F=labels",
                "",
                "Analysis:",
                "G=Generate all graphs",
                "J=Save current data",
                "H=Help"
            ]
            self.ui.panel(self.screen, control_lines, Config.WIDTH - 220, Config.HEIGHT - 180)

            # Show analysis status
            if self.analysis_status and (current_time - self.analysis_timer) < 10000:  # Show for 10 seconds
                status_lines = [
                    "Analysis Status:",
                    self.analysis_status[:50] + "..." if len(self.analysis_status) > 50 else self.analysis_status
                ]
                self.ui.panel(self.screen, status_lines, 20, Config.HEIGHT - 100, (20, 40, 20))

            if self.show_help:
                self.ui.help_overlay(self.screen)

            pygame.display.flip()


# --------------------
# Analysis Usage Example
# --------------------
def run_analysis_only():
    """Run analysis without the interactive simulation"""
    print("Running comprehensive locomotion analysis...")
    analyzer = DataAnalyzer()
    analyzer.create_all_graphs()
    analyzer.save_data_to_csv()
    print("Analysis complete! Check the 'analysis_graphs' folder for results.")


if __name__ == "__main__":
    # You can run either the interactive game or analysis only
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--analysis-only":
        run_analysis_only()
    else:
        print("Starting interactive simulation...")
        print("Press 'G' in the simulation to generate comprehensive analysis graphs")
        print("Press 'J' to save current simulation data")
        print("Or run with --analysis-only flag to generate all graphs without simulation")
        Game().run()