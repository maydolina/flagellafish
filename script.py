"""
Simple runner script for the locomotion analysis.
Just run this file to generate all graphs!

Usage: python run_analysis.py
"""

# Copy the main analysis code here and run it
import sys
import os

# Add the analysis code
exec(open('plotting.py').read())

print("=" * 60)
print("LOCOMOTION ANALYSIS COMPLETE!")
print("=" * 60)
print(f"All graphs have been saved to: {os.path.abspath('locomotion_analysis_plots')}")
print("\nGenerated 14 comprehensive plots:")
print("1. Speed vs Viscosity - Shows how fluid thickness affects speed")
print("2. Speed vs Reynolds Number - Performance across flow regimes")
print("3. Efficiency vs Reynolds Number - When each propulsion works best")
print("4. Speed vs Frequency - Optimal stroke rates")
print("5. Speed vs Size - How organism size affects performance")
print("6. Reynolds vs Size - Flow regimes for different organisms")
print("7. Thrust vs Drag Analysis - Force balance breakdown")
print("8. Environment Comparison - Performance in different fluids")
print("9. Optimal Frequency - Best stroke rates for each size")
print("10. Performance Heatmaps - 2D performance landscapes")
print("11. Power Analysis - Energy consumption patterns")
print("12. Efficiency Landscape - 2D efficiency maps")
print("13. Regime Transitions - Flow regime boundaries")
print("14. 3D Surfaces - Three-dimensional performance visualization")
print("\nOpen the 'locomotion_analysis_plots' folder to view all graphs!")