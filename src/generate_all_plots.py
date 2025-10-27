#!/usr/bin/env python3
"""
Master visualization script for PINN heat sink project.
Generates all 5 key visualizations for project presentation.
"""

import sys
from pathlib import Path

def run_all_visualizations():
    """Run all visualization scripts in sequence."""
    
    print("=" * 60)
    print("PINN Heat Sink Project - Visualization Suite")
    print("=" * 60)
    
    # Ensure plots directory exists
    Path("plots").mkdir(exist_ok=True)
    
    visualization_scripts = [
        ("1. Ground Truth Visualization", "src/plot_ground_truth.py"),
        ("2. Workflow Comparison Diagram", "src/plot_workflow_comparison.py"),
        ("3. Prediction vs Ground Truth", "src/plot_prediction_vs_truth.py"),
        ("4. Absolute Error Map", "src/plot_error_map.py"),
        ("5. Training Loss Curve", "src/plot_loss_curve.py")
    ]
    
    successful_plots = []
    failed_plots = []
    
    for description, script_path in visualization_scripts:
        print(f"\n{description}...")
        try:
            # Import and run the script
            exec(open(script_path).read())
            successful_plots.append(description)
            print(f"✅ {description} completed successfully")
        except Exception as e:
            print(f"❌ {description} failed: {str(e)}")
            failed_plots.append((description, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    
    print(f"✅ Successful: {len(successful_plots)}")
    for plot in successful_plots:
        print(f"   - {plot}")
    
    if failed_plots:
        print(f"\n❌ Failed: {len(failed_plots)}")
        for plot, error in failed_plots:
            print(f"   - {plot}: {error}")
    
    print(f"\n📁 All plots saved to: plots/")
    print("\nGenerated files:")
    plot_files = list(Path("plots").glob("*.png"))
    for file in sorted(plot_files):
        print(f"   - {file}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Download the generated .png files from the plots/ directory")
    print("2. Include them in your project report/presentation")
    print("3. Use the workflow comparison diagram to explain the value proposition")
    print("4. Use the error maps to highlight current challenges and future work")
    print("=" * 60)

if __name__ == "__main__":
    run_all_visualizations()
