"""
Debug version that shows exactly what's happening step by step.
This will help us figure out why the folder isn't being created.
"""

import os
import sys
import traceback

print("=== LOCOMOTION ANALYSIS DEBUG MODE ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__ if '__file__' in globals() else 'Interactive mode'}")

# Test 1: Can we create directories?
print("\n--- TEST 1: Directory Creation ---")
try:
    test_dir = "test_directory_creation"
    os.makedirs(test_dir, exist_ok=True)
    print(f"✅ Successfully created directory: {test_dir}")
    print(f"✅ Full path: {os.path.abspath(test_dir)}")
    print(f"✅ Directory exists: {os.path.exists(test_dir)}")
    print(f"✅ Directory contents: {os.listdir('.')}")
except Exception as e:
    print(f"❌ Failed to create directory: {e}")
    traceback.print_exc()

# Test 2: Can we import required libraries?
print("\n--- TEST 2: Library Import ---")
try:
    import matplotlib.pyplot as plt

    print("✅ matplotlib imported successfully")
except ImportError as e:
    print(f"❌ matplotlib import failed: {e}")
    print("Run: pip install matplotlib")

try:
    import numpy as np

    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")
    print("Run: pip install numpy")

# Test 3: Simple plot creation
print("\n--- TEST 3: Plot Creation ---")
try:
    import matplotlib.pyplot as plt
    import numpy as np

    # Create the analysis directory
    output_dir = "locomotion_analysis_plots"
    print(f"Attempting to create directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Directory created: {os.path.abspath(output_dir)}")

    # Create a simple test plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Test Plot - Sin Wave')
    plt.grid(True)

    # Save the plot
    test_file = os.path.join(output_dir, "test_plot.png")
    plt.savefig(test_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Test plot saved to: {test_file}")
    print(f"✅ File exists: {os.path.exists(test_file)}")
    print(f"✅ File size: {os.path.getsize(test_file) if os.path.exists(test_file) else 'N/A'} bytes")

except Exception as e:
    print(f"❌ Plot creation failed: {e}")
    traceback.print_exc()

# Test 4: List all files and directories
print("\n--- TEST 4: Directory Listing ---")
try:
    current_files = os.listdir('.')
    print("Files and directories in current location:")
    for item in current_files:
        if os.path.isdir(item):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")

    # Check if our directory exists
    if "locomotion_analysis_plots" in current_files:
        print(f"\n✅ Found locomotion_analysis_plots directory!")
        plot_files = os.listdir("locomotion_analysis_plots")
        print(f"Contents of locomotion_analysis_plots:")
        for file in plot_files:
            print(f"  📊 {file}")
    else:
        print(f"\n❌ locomotion_analysis_plots directory NOT found")

except Exception as e:
    print(f"❌ Directory listing failed: {e}")

# Test 5: Create a minimal working example
print("\n--- TEST 5: Minimal Working Example ---")
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import math


    # Physics functions
    def reynolds(rho, U, L, mu):
        if mu <= 0 or L <= 0:
            return 0.0
        return rho * U * L / mu


    def fin_eff(Re):
        x = math.log10(max(Re, 1e-6))
        t = 1 / (1 + math.exp(-(x - 0.5) * 2.5))
        return max(0.02, min(1.0, t))


    def flag_eff(Re):
        x = math.log10(max(Re, 1e-6))
        t = 1 / (1 + math.exp((x - 0.0) * 2.5))
        return max(0.05, min(1.2, t * 1.2))


    # Create directory
    output_dir = "locomotion_analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory created at: {os.path.abspath(output_dir)}")

    # Generate simple viscosity plot
    plt.figure(figsize=(10, 6))

    mu_values = np.logspace(-4, 1, 50)
    speeds_fin = []
    speeds_flag = []

    for mu in mu_values:
        # Simplified calculation
        Re = reynolds(1000, 0.01, 1e-4, mu)
        eff_fin = fin_eff(Re)
        eff_flag = flag_eff(Re)

        speed_fin = eff_fin * 0.01 / (1 + mu * 1000)
        speed_flag = eff_flag * 0.01 / (1 + mu * 1000)

        speeds_fin.append(speed_fin)
        speeds_flag.append(speed_flag)

    plt.loglog(mu_values, speeds_fin, 'b-', linewidth=3, label='Fins', marker='o')
    plt.loglog(mu_values, speeds_flag, 'r-', linewidth=3, label='Flagella', marker='s')
    plt.xlabel('Viscosity (Pa·s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed vs Viscosity - Debug Test')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    filename = os.path.join(output_dir, "debug_speed_vs_viscosity.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Plot saved successfully!")
    print(f"✅ Filename: {filename}")
    print(f"✅ File exists: {os.path.exists(filename)}")
    print(f"✅ File size: {os.path.getsize(filename)} bytes")

    # Try to open the directory
    import subprocess, platform

    try:
        if platform.system() == "Windows":
            subprocess.run(f'explorer "{os.path.abspath(output_dir)}"', shell=True)
            print("✅ Opened folder in Windows Explorer")
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", os.path.abspath(output_dir)])
            print("✅ Opened folder in Finder")
        else:  # Linux
            subprocess.run(["xdg-open", os.path.abspath(output_dir)])
            print("✅ Opened folder in file manager")
    except Exception as e:
        print(f"⚠️ Could not auto-open folder: {e}")
        print(f"Manual path: {os.path.abspath(output_dir)}")

except Exception as e:
    print(f"❌ Minimal example failed: {e}")
    traceback.print_exc()

print("\n=== DEBUG COMPLETE ===")
print(f"If successful, check: {os.path.abspath('locomotion_analysis_plots')}")
print("Look for a file called 'debug_speed_vs_viscosity.png'")