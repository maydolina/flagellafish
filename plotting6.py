import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import os


class Propulsion(Enum):
    FIN = "Fin"
    FLAGELLA = "Flagella"


class RealisticLocomotionAnalyzer:
    def __init__(self):
        self.results = []

    def reynolds_number(self, velocity, length, kinematic_viscosity):
        """Calculate Reynolds number"""
        return velocity * length / kinematic_viscosity

    def drag_coefficient(self, reynolds):
        """Realistic drag coefficient based on Reynolds number"""
        if reynolds < 0.1:
            return 24 / max(reynolds, 0.01)  # Stokes flow
        elif reynolds < 1000:
            return 24 / reynolds + 6 / (1 + np.sqrt(reynolds)) + 0.4
        else:
            return 0.4  # Turbulent flow

    def propulsion_efficiency(self, reynolds, propulsion_type):
        """More realistic efficiency based on actual research"""
        if propulsion_type == Propulsion.FLAGELLA:
            # Flagella work better at low Re
            if reynolds < 0.001:
                return 0.15
            elif reynolds < 0.1:
                return 0.1 * (0.1 / reynolds) ** 0.3
            else:
                return 0.02  # Very inefficient at high Re

        else:  # Fins
            # Fins work better at moderate to high Re
            if reynolds < 0.1:
                return 0.05 * reynolds  # Poor at low Re
            elif reynolds < 100:
                return 0.3 * (reynolds / 10) ** 0.2
            else:
                return 0.7  # Good at high Re

    def calculate_steady_state_velocity(self, length, frequency, density,
                                        kinematic_viscosity, propulsion_type):
        """Calculate steady-state swimming velocity using realistic physics"""

        # Initial guess for velocity
        velocity = 0.01

        for iteration in range(100):  # Iterative solution
            re = self.reynolds_number(velocity, length, kinematic_viscosity)

            # Calculate thrust based on propulsion type
            if propulsion_type == Propulsion.FLAGELLA:
                # Flagellar thrust (based on resistive force theory)
                amplitude = 0.1 * length
                thrust_coeff = 2 * np.pi * (density * kinematic_viscosity) / np.log(0.18 * length / (0.001 * length))
                thrust = thrust_coeff * frequency * amplitude
            else:
                # Fin thrust (based on momentum theory)
                fin_area = 0.2 * length ** 2
                stroke_velocity = frequency * 0.3 * length
                thrust = 0.5 * density * fin_area * stroke_velocity ** 2 * self.propulsion_efficiency(re,
                                                                                                      propulsion_type)

            # Calculate drag
            frontal_area = np.pi * (length / 2) ** 2
            drag = 0.5 * density * velocity ** 2 * frontal_area * self.drag_coefficient(re)

            # Update velocity based on force balance
            new_velocity = (thrust / (0.5 * density * frontal_area * self.drag_coefficient(re))) ** 0.5

            # Check convergence
            if abs(new_velocity - velocity) / velocity < 0.001:
                break

            velocity = 0.7 * velocity + 0.3 * new_velocity  # Damped iteration

        return velocity, re, self.propulsion_efficiency(re, propulsion_type)

    def generate_realistic_data(self):
        """Generate data with realistic parameter ranges"""

        # Realistic organism sizes
        bacteria_sizes = np.logspace(-6, -4.5, 15)  # 1-30 μm
        fish_sizes = np.logspace(-2, 0, 15)  # 1 cm - 1 m

        # Realistic frequencies
        bacteria_freq = np.logspace(1, 3, 10)  # 10-1000 Hz
        fish_freq = np.logspace(-0.5, 1.5, 10)  # 0.3-30 Hz

        # Real fluid properties
        fluids = {
            'Water': {'density': 1000, 'kinematic_viscosity': 1e-6},
            'Honey': {'density': 1400, 'kinematic_viscosity': 1e-2},
            'Air': {'density': 1.2, 'kinematic_viscosity': 1.5e-5}
        }

        print("Generating realistic locomotion data...")

        for fluid_name, props in fluids.items():
            density = props['density']
            nu = props['kinematic_viscosity']

            # Bacteria data
            for size in bacteria_sizes:
                for freq in bacteria_freq:
                    for propulsion in [Propulsion.FIN, Propulsion.FLAGELLA]:
                        try:
                            velocity, re, efficiency = self.calculate_steady_state_velocity(
                                size, freq, density, nu, propulsion)

                            self.results.append({
                                'organism': 'bacteria',
                                'length': size,
                                'frequency': freq,
                                'fluid': fluid_name,
                                'density': density,
                                'kinematic_viscosity': nu,
                                'propulsion': propulsion.value,
                                'velocity': velocity,
                                'reynolds': re,
                                'efficiency': efficiency,
                                'power': velocity ** 3 * density * size ** 2  # Rough power estimate
                            })
                        except:
                            continue

            # Fish data
            for size in fish_sizes:
                for freq in fish_freq:
                    for propulsion in [Propulsion.FIN, Propulsion.FLAGELLA]:
                        try:
                            velocity, re, efficiency = self.calculate_steady_state_velocity(
                                size, freq, density, nu, propulsion)

                            self.results.append({
                                'organism': 'fish',
                                'length': size,
                                'frequency': freq,
                                'fluid': fluid_name,
                                'density': density,
                                'kinematic_viscosity': nu,
                                'propulsion': propulsion.value,
                                'velocity': velocity,
                                'reynolds': re,
                                'efficiency': efficiency,
                                'power': velocity ** 3 * density * size ** 2
                            })
                        except:
                            continue

        print(f"Generated {len(self.results)} data points")

    def create_meaningful_plots(self):
        """Create plots that actually make sense"""

        if not self.results:
            self.generate_realistic_data()

        df = pd.DataFrame(self.results)

        # Create output directory
        os.makedirs('realistic_analysis', exist_ok=True)

        # Set plotting style
        plt.style.use('dark_background')

        # 1. Reynolds Number vs Efficiency (the key relationship)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Realistic Locomotion Analysis', fontsize=16, color='white')

        # Plot efficiency vs Reynolds for each propulsion type
        for propulsion in ['Fin', 'Flagella']:
            data = df[df['propulsion'] == propulsion]
            color = '#00b4ff' if propulsion == 'Fin' else '#ffb400'
            ax1.semilogx(data['reynolds'], data['efficiency'], 'o',
                         alpha=0.6, color=color, label=propulsion, markersize=4)

        ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Re = 1')
        ax1.set_xlabel('Reynolds Number')
        ax1.set_ylabel('Propulsion Efficiency')
        ax1.set_title('Efficiency vs Reynolds Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Speed vs Size for different fluids
        fluids = df['fluid'].unique()
        colors = ['#6496ff', '#ffb400', '#ff6b6b']

        for i, fluid in enumerate(fluids):
            fluid_data = df[df['fluid'] == fluid]
            ax2.loglog(fluid_data['length'], fluid_data['velocity'], 'o',
                       alpha=0.6, color=colors[i], label=fluid, markersize=3)

        ax2.set_xlabel('Organism Length (m)')
        ax2.set_ylabel('Swimming Speed (m/s)')
        ax2.set_title('Speed vs Size in Different Fluids')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Propulsion comparison by organism type
        organisms = ['bacteria', 'fish']
        propulsion_types = ['Fin', 'Flagella']

        speeds = []
        labels = []
        colors_bar = []

        for org in organisms:
            for prop in propulsion_types:
                data = df[(df['organism'] == org) & (df['propulsion'] == prop)]
                if len(data) > 0:
                    speeds.append(data['velocity'].mean())
                    labels.append(f'{org.title()}\n{prop}')
                    colors_bar.append('#00b4ff' if prop == 'Fin' else '#ffb400')

        bars = ax3.bar(labels, speeds, color=colors_bar, alpha=0.8)
        ax3.set_ylabel('Average Speed (m/s)')
        ax3.set_title('Average Speed by Organism and Propulsion Type')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # 4. Optimal frequency vs size
        size_bins = np.logspace(-6, 0, 20)
        optimal_frequencies = []
        bin_centers = []

        for i in range(len(size_bins) - 1):
            size_mask = (df['length'] >= size_bins[i]) & (df['length'] < size_bins[i + 1])
            subset = df[size_mask]

            if len(subset) > 0:
                # Find frequency that gives maximum speed for each size bin
                best_idx = subset['velocity'].idxmax()
                optimal_frequencies.append(subset.loc[best_idx, 'frequency'])
                bin_centers.append((size_bins[i] + size_bins[i + 1]) / 2)

        ax4.loglog(bin_centers, optimal_frequencies, 'o-', color='#ff6b6b', markersize=6)
        ax4.set_xlabel('Organism Size (m)')
        ax4.set_ylabel('Optimal Frequency (Hz)')
        ax4.set_title('Optimal Swimming Frequency vs Size')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('realistic_analysis/realistic_locomotion_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Save data
        df.to_csv('realistic_analysis/realistic_data.csv', index=False)

        print("Realistic analysis complete! Check 'realistic_analysis' folder.")


# Run the analysis
if __name__ == "__main__":
    analyzer = RealisticLocomotionAnalyzer()
    analyzer.create_meaningful_plots()