"""
Comprehensive Data Analysis Tool for Locomotion Demo
Generates multiple graphs comparing fin vs flagella performance across different parameters.

Run: pip install pygame matplotlib numpy pandas seaborn
"""

import pygame
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Tuple, List, Dict
import os
from dataclasses import dataclass
from enum import Enum

# Set up matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# Copy the physics functions from the original code
def reynolds(rho: float, U: float, L: float, mu: float) -> float:
    if mu <= 0 or L <= 0:
        return 0.0
    return rho * U * L / mu


def cd_blend(Re: float) -> float:
    """Blend low-Re (Stokes) and quadratic drag. Returns a nominal Cd for the quadratic term."""
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


class PropulsionType(Enum):
    FIN = "Fin"
    FLAGELLA = "Flagella"


class LocomotionAnalyzer:
    def __init__(self):
        self.environments = {
            "Water": {"rho": 1000.0, "mu": 0.001},
            "Honey": {"rho": 1400.0, "mu": 10.0},
            "Air": {"rho": 1.2, "mu": 0.000018},
            "Oil": {"rho": 900.0, "mu": 0.1},
            "Cytoplasm": {"rho": 1100.0, "mu": 0.01}
        }

        # Create output directory
        self.output_dir = "locomotion_analysis_plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_steady_state_speed(self, L: float, freq: float, prop_type: PropulsionType,
                                     rho: float, mu: float, max_iterations: int = 1000) -> Tuple[float, float, float]:
        """Calculate steady-state speed using iterative method"""
        U = 0.1  # Initial guess
        dt = 0.001

        for _ in range(max_iterations):
            Re = reynolds(rho, abs(U), L, mu)

            # Calculate thrust
            if prop_type == PropulsionType.FIN:
                eff = fin_eff(Re)
                amp = 0.2 * L
                thrust = 0.5 * rho * (freq * amp) ** 2 * L ** 2 * eff
            else:
                eff = flag_eff(Re)
                pitch = 0.2 * L
                thrust = 6 * math.pi * mu * (L / 2) * (freq * pitch) * eff

            # Calculate drag
            r = L / 2
            A = math.pi * r * r
            Cd = cd_blend(max(Re, 1e-6))
            drag_linear = 6 * math.pi * mu * r * abs(U)
            drag_quad = 0.5 * rho * Cd * A * U * U
            drag = drag_linear + drag_quad

            # Update velocity
            mass = max(1e-9, rho * (4 / 3) * math.pi * r ** 3)
            net = thrust - drag
            ax = net / mass
            U_new = max(0.0, U + ax * dt)

            if abs(U_new - U) < 1e-8:
                break
            U = U_new

        Re_final = reynolds(rho, abs(U), L, mu)
        return U, Re_final, eff

    def generate_all_plots(self):
        """Generate all analysis plots"""
        print("Generating comprehensive locomotion analysis plots...")

        # 1. Speed vs Viscosity (your requested plot)
        self.plot_speed_vs_viscosity()

        # 2. Speed vs Reynolds Number
        self.plot_speed_vs_reynolds()

        # 3. Efficiency vs Reynolds Number
        self.plot_efficiency_vs_reynolds()

        # 4. Speed vs Frequency
        self.plot_speed_vs_frequency()

        # 5. Speed vs Size
        self.plot_speed_vs_size()

        # 6. Reynolds vs Size for different environments
        self.plot_reynolds_vs_size()

        # 7. Thrust vs Drag analysis
        self.plot_thrust_vs_drag()

        # 8. Performance in different environments
        self.plot_environment_comparison()

        # 9. Optimal frequency analysis
        self.plot_optimal_frequency()

        # 10. Size-frequency performance maps
        #self.plot_performance_heatmaps()

        # 11. Power consumption analysis
        #self.plot_power_analysis()

        # 12. Efficiency landscape
        #self.plot_efficiency_landscape()

        # 13. Regime transition plots
        #self.plot_regime_transitions()

        # 14. 3D performance surfaces
        #self.plot_3d_surfaces()

        print(f"All plots saved to {self.output_dir}/")

    def plot_speed_vs_viscosity(self):
        """Plot speed vs viscosity for both propulsion types"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Viscosity range (log scale)
        mu_values = np.logspace(-6, 2, 100)  # 1e-6 to 100 Pa·s
        rho = 1000.0  # Water density

        scenarios = [
            {"L": 5e-6, "freq": 100, "title": "Bacteria Scale (5μm, 100Hz)"},
            {"L": 1e-4, "freq": 50, "title": "Small Microorganism (100μm, 50Hz)"},
            {"L": 1e-2, "freq": 10, "title": "Large Microorganism (1cm, 10Hz)"},
            {"L": 0.1, "freq": 4, "title": "Fish Scale (10cm, 4Hz)"}
        ]

        axes = [ax1, ax2, ax3, ax4]

        for i, (ax, scenario) in enumerate(zip(axes, scenarios)):
            speeds_fin = []
            speeds_flag = []

            for mu in mu_values:
                speed_fin, _, _ = self.calculate_steady_state_speed(
                    scenario["L"], scenario["freq"], PropulsionType.FIN, rho, mu)
                speed_flag, _, _ = self.calculate_steady_state_speed(
                    scenario["L"], scenario["freq"], PropulsionType.FLAGELLA, rho, mu)

                speeds_fin.append(speed_fin)
                speeds_flag.append(speed_flag)

            ax.loglog(mu_values, speeds_fin, 'b-', linewidth=2, label='Fins', alpha=0.8)
            ax.loglog(mu_values, speeds_flag, 'r-', linewidth=2, label='Flagella', alpha=0.8)
            ax.set_xlabel('Viscosity (Pa·s)')
            ax.set_ylabel('Speed (m/s)')
            ax.set_title(scenario["title"])
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/speed_vs_viscosity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_speed_vs_reynolds(self):
        """Plot speed vs Reynolds number"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Generate data across different sizes and environments
        Re_values = np.logspace(-3, 5, 100)

        for env_name, env_props in self.environments.items():
            speeds_fin = []
            speeds_flag = []

            for Re in Re_values:
                # For a given Re, we need to find L and U that satisfy Re = rho*U*L/mu
                # Let's fix L and solve for U
                L = 1e-4  # Fixed characteristic length
                U_target = Re * env_props["mu"] / (env_props["rho"] * L)

                # Use this as initial guess and refine
                speed_fin, Re_actual_fin, _ = self.calculate_steady_state_speed(
                    L, 50, PropulsionType.FIN, env_props["rho"], env_props["mu"])
                speed_flag, Re_actual_flag, _ = self.calculate_steady_state_speed(
                    L, 50, PropulsionType.FLAGELLA, env_props["rho"], env_props["mu"])

                speeds_fin.append(speed_fin)
                speeds_flag.append(speed_flag)

            ax1.loglog(Re_values, speeds_fin, linewidth=2, label=f'Fins - {env_name}', alpha=0.7)
            ax2.loglog(Re_values, speeds_flag, linewidth=2, label=f'Flagella - {env_name}', alpha=0.7)

        ax1.set_xlabel('Reynolds Number')
        ax1.set_ylabel('Speed (m/s)')
        ax1.set_title('Fin Propulsion vs Reynolds Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Reynolds Number')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Flagella Propulsion vs Reynolds Number')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/speed_vs_reynolds.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_efficiency_vs_reynolds(self):
        """Plot efficiency vs Reynolds number"""
        Re_values = np.logspace(-3, 5, 1000)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        fin_efficiencies = [fin_eff(Re) for Re in Re_values]
        flag_efficiencies = [flag_eff(Re) for Re in Re_values]

        ax1.semilogx(Re_values, fin_efficiencies, 'b-', linewidth=3, label='Fin Efficiency')
        ax1.semilogx(Re_values, flag_efficiencies, 'r-', linewidth=3, label='Flagella Efficiency')
        ax1.set_xlabel('Reynolds Number')
        ax1.set_ylabel('Efficiency')
        ax1.set_title('Propulsion Efficiency vs Reynolds Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='Re = 1')
        ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='Re = 100')

        # Ratio plot
        efficiency_ratio = np.array(flag_efficiencies) / np.array(fin_efficiencies)
        ax2.semilogx(Re_values, efficiency_ratio, 'g-', linewidth=3)
        ax2.axhline(y=1, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(Re_values, efficiency_ratio, 1,
                         where=(efficiency_ratio > 1), alpha=0.3, color='red',
                         label='Flagella Better')
        ax2.fill_between(Re_values, efficiency_ratio, 1,
                         where=(efficiency_ratio < 1), alpha=0.3, color='blue',
                         label='Fins Better')
        ax2.set_xlabel('Reynolds Number')
        ax2.set_ylabel('Efficiency Ratio (Flagella/Fins)')
        ax2.set_title('Relative Efficiency: Flagella vs Fins')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/efficiency_vs_reynolds.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_speed_vs_frequency(self):
        """Plot speed vs frequency for different sizes"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        frequencies = np.logspace(0, 3, 50)  # 1 to 1000 Hz
        sizes = [5e-6, 5e-5, 5e-4, 5e-3]  # Different organism sizes
        size_labels = ["5μm (Bacteria)", "50μm (Large Bacteria)", "500μm (Protozoa)", "5mm (Small Fish)"]

        rho, mu = 1000.0, 0.001  # Water properties

        axes = [ax1, ax2, ax3, ax4]

        for i, (ax, L, label) in enumerate(zip(axes, sizes, size_labels)):
            speeds_fin = []
            speeds_flag = []

            for freq in frequencies:
                speed_fin, _, _ = self.calculate_steady_state_speed(L, freq, PropulsionType.FIN, rho, mu)
                speed_flag, _, _ = self.calculate_steady_state_speed(L, freq, PropulsionType.FLAGELLA, rho, mu)

                speeds_fin.append(speed_fin)
                speeds_flag.append(speed_flag)

            ax.loglog(frequencies, speeds_fin, 'b-', linewidth=2, label='Fins', alpha=0.8)
            ax.loglog(frequencies, speeds_flag, 'r-', linewidth=2, label='Flagella', alpha=0.8)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Speed (m/s)')
            ax.set_title(f'Speed vs Frequency - {label}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/speed_vs_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_speed_vs_size(self):
        """Plot speed vs organism size"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        sizes = np.logspace(-6, -1, 100)  # 1μm to 10cm
        frequencies = [10, 50, 100, 500]  # Different frequencies

        rho, mu = 1000.0, 0.001  # Water properties

        axes = [ax1, ax2, ax3, ax4]

        for i, (ax, freq) in enumerate(zip(axes, frequencies)):
            speeds_fin = []
            speeds_flag = []

            for L in sizes:
                speed_fin, _, _ = self.calculate_steady_state_speed(L, freq, PropulsionType.FIN, rho, mu)
                speed_flag, _, _ = self.calculate_steady_state_speed(L, freq, PropulsionType.FLAGELLA, rho, mu)

                speeds_fin.append(speed_fin)
                speeds_flag.append(speed_flag)

            ax.loglog(sizes * 1e6, speeds_fin, 'b-', linewidth=2, label='Fins', alpha=0.8)
            ax.loglog(sizes * 1e6, speeds_flag, 'r-', linewidth=2, label='Flagella', alpha=0.8)
            ax.set_xlabel('Size (μm)')
            ax.set_ylabel('Speed (m/s)')
            ax.set_title(f'Speed vs Size - {freq} Hz')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/speed_vs_size.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_reynolds_vs_size(self):
        """Plot Reynolds number vs size for different environments"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sizes = np.logspace(-6, -1, 100)
        freq = 50  # Fixed frequency

        for env_name, env_props in self.environments.items():
            Re_fins = []
            Re_flags = []

            for L in sizes:
                _, Re_fin, _ = self.calculate_steady_state_speed(L, freq, PropulsionType.FIN,
                                                                 env_props["rho"], env_props["mu"])
                _, Re_flag, _ = self.calculate_steady_state_speed(L, freq, PropulsionType.FLAGELLA,
                                                                  env_props["rho"], env_props["mu"])
                Re_fins.append(Re_fin)
                Re_flags.append(Re_flag)

            ax1.loglog(sizes * 1e6, Re_fins, linewidth=2, label=f'Fins - {env_name}', alpha=0.7)
            ax2.loglog(sizes * 1e6, Re_flags, linewidth=2, label=f'Flagella - {env_name}', alpha=0.7)

        for ax in [ax1, ax2]:
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Re = 1')
            ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Re = 100')
            ax.set_xlabel('Size (μm)')
            ax.set_ylabel('Reynolds Number')
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax1.set_title('Reynolds Number vs Size - Fins')
        ax2.set_title('Reynolds Number vs Size - Flagella')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/reynolds_vs_size.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_thrust_vs_drag(self):
        """Analyze thrust and drag components"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        speeds = np.logspace(-4, 1, 100)  # Range of speeds
        L = 1e-4  # Fixed size
        freq = 50  # Fixed frequency
        rho, mu = 1000.0, 0.001

        # Calculate forces for different speeds
        thrust_fin = []
        thrust_flag = []
        drag_linear = []
        drag_quad = []
        drag_total = []

        for U in speeds:
            Re = reynolds(rho, U, L, mu)

            # Thrust calculations
            eff_fin = fin_eff(Re)
            eff_flag = flag_eff(Re)
            amp = 0.2 * L
            pitch = 0.2 * L

            T_fin = 0.5 * rho * (freq * amp) ** 2 * L ** 2 * eff_fin
            T_flag = 6 * math.pi * mu * (L / 2) * (freq * pitch) * eff_flag

            # Drag calculations
            r = L / 2
            A = math.pi * r * r
            Cd = cd_blend(max(Re, 1e-6))
            D_linear = 6 * math.pi * mu * r * U
            D_quad = 0.5 * rho * Cd * A * U * U
            D_total = D_linear + D_quad

            thrust_fin.append(T_fin)
            thrust_flag.append(T_flag)
            drag_linear.append(D_linear)
            drag_quad.append(D_quad)
            drag_total.append(D_total)

        # Plot thrust vs speed
        ax1.loglog(speeds, thrust_fin, 'b-', linewidth=2, label='Fin Thrust')
        ax1.loglog(speeds, thrust_flag, 'r-', linewidth=2, label='Flagella Thrust')
        ax1.loglog(speeds, drag_total, 'k--', linewidth=2, label='Total Drag')
        ax1.set_xlabel('Speed (m/s)')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Thrust and Drag vs Speed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot drag components
        ax2.loglog(speeds, drag_linear, 'g-', linewidth=2, label='Linear Drag (Stokes)')
        ax2.loglog(speeds, drag_quad, 'm-', linewidth=2, label='Quadratic Drag')
        ax2.loglog(speeds, drag_total, 'k-', linewidth=2, label='Total Drag')
        ax2.set_xlabel('Speed (m/s)')
        ax2.set_ylabel('Drag Force (N)')
        ax2.set_title('Drag Components vs Speed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot power vs speed
        power_fin = np.array(thrust_fin) * speeds
        power_flag = np.array(thrust_flag) * speeds
        power_drag = np.array(drag_total) * speeds

        ax3.loglog(speeds, power_fin, 'b-', linewidth=2, label='Fin Power')
        ax3.loglog(speeds, power_flag, 'r-', linewidth=2, label='Flagella Power')
        ax3.loglog(speeds, power_drag, 'k--', linewidth=2, label='Drag Power')
        ax3.set_xlabel('Speed (m/s)')
        ax3.set_ylabel('Power (W)')
        ax3.set_title('Power vs Speed')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot efficiency vs speed
        Re_values = [reynolds(rho, U, L, mu) for U in speeds]
        eff_fin_values = [fin_eff(Re) for Re in Re_values]
        eff_flag_values = [flag_eff(Re) for Re in Re_values]

        ax4.semilogx(speeds, eff_fin_values, 'b-', linewidth=2, label='Fin Efficiency')
        ax4.semilogx(speeds, eff_flag_values, 'r-', linewidth=2, label='Flagella Efficiency')
        ax4.set_xlabel('Speed (m/s)')
        ax4.set_ylabel('Efficiency')
        ax4.set_title('Efficiency vs Speed')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/thrust_vs_drag_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_environment_comparison(self):
        """Compare performance across all environments"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Fixed parameters
        sizes = [5e-6, 1e-4, 1e-2, 0.1]
        size_labels = ["Bacteria", "Large Bacteria", "Protozoa", "Fish"]
        freq = 50

        env_names = list(self.environments.keys())

        # Speed comparison
        speeds_fin = {env: [] for env in env_names}
        speeds_flag = {env: [] for env in env_names}

        for env_name, env_props in self.environments.items():
            for L in sizes:
                speed_fin, _, _ = self.calculate_steady_state_speed(L, freq, PropulsionType.FIN,
                                                                    env_props["rho"], env_props["mu"])
                speed_flag, _, _ = self.calculate_steady_state_speed(L, freq, PropulsionType.FLAGELLA,
                                                                     env_props["rho"], env_props["mu"])
                speeds_fin[env_name].append(speed_fin)
                speeds_flag[env_name].append(speed_flag)

        # Bar plot comparison
        x = np.arange(len(size_labels))
        width = 0.15

        for i, env in enumerate(env_names):
            ax1.bar(x + i * width, speeds_fin[env], width, label=f'Fins - {env}', alpha=0.8)
            ax2.bar(x + i * width, speeds_flag[env], width, label=f'Flagella - {env}', alpha=0.8)

        ax1.set_xlabel('Organism Size')
        ax1.set_ylabel('Speed (m/s)')
        ax1.set_title('Fin Speed Across Environments')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(size_labels)
        ax1.legend()
        ax1.set_yscale('log')

        ax2.set_xlabel('Organism Size')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Flagella Speed Across Environments')
        ax2.set_xticks(x + width * 2)
        ax2.set_xticklabels(size_labels)
        ax2.legend()
        ax2.set_yscale('log')

        # Viscosity vs performance
        mu_values = [self.environments[env]["mu"] for env in env_names]
        L_test = 1e-4

        speeds_fin_vs_mu = []
        speeds_flag_vs_mu = []

        for env_name, env_props in self.environments.items():
            speed_fin, _, _ = self.calculate_steady_state_speed(L_test, freq, PropulsionType.FIN,
                                                                env_props["rho"], env_props["mu"])
            speed_flag, _, _ = self.calculate_steady_state_speed(L_test, freq, PropulsionType.FLAGELLA,
                                                                 env_props["rho"], env_props["mu"])
            speeds_fin_vs_mu.append(speed_fin)
            speeds_flag_vs_mu.append(speed_flag)

        ax3.loglog(mu_values, speeds_fin_vs_mu, 'bo-', linewidth=2, markersize=8, label='Fins')
        ax3.loglog(mu_values, speeds_flag_vs_mu, 'ro-', linewidth=2, markersize=8, label='Flagella')

        for i, env in enumerate(env_names):
            ax3.annotate(env, (mu_values[i], speeds_fin_vs_mu[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax3.set_xlabel('Viscosity (Pa·s)')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Speed vs Environment Viscosity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Winner analysis
        winners = []
        for env_name in env_names:
            fin_speed = speeds_fin_vs_mu[env_names.index(env_name)]
            flag_speed = speeds_flag_vs_mu[env_names.index(env_name)]
            winners.append('Flagella' if flag_speed > fin_speed else 'Fins')

        colors = ['blue' if w == 'Fins' else 'red' for w in winners]
        ax4.bar(env_names, [1] * len(env_names), color=colors, alpha=0.6)
        ax4.set_ylabel('Winner')
        ax4.set_title('Optimal Propulsion by Environment')
        ax4.set_ylim(0, 1.2)

        for i, (env, winner) in enumerate(zip(env_names, winners)):
            ax4.text(i, 0.5, winner, ha='center', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/environment_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_optimal_frequency(self):
        """Find optimal frequencies for different sizes and environments"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        frequencies = np.logspace(0, 3, 50)
        sizes = [5e-6, 5e-5, 5e-4, 5e-3]
        size_labels = ["5μm", "50μm", "500μm", "5mm"]

        # For water
        rho, mu = 1000.0, 0.001

        axes = [ax1, ax2, ax3, ax4]

        for i, (ax, L, label) in enumerate(zip(axes, sizes, size_labels)):
            speeds_fin = []

if __name__ == "__main__":
    analyzer = LocomotionAnalyzer()
    analyzer.generate_all_plots()
