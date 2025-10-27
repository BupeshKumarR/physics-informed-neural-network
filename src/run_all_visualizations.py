"""
Master Visualization Runner
Generates all key visualizations for different audiences
"""

import sys
from pathlib import Path

# Add src directory to path, and parent (root) for config imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_all_visualizations():
    """Run all visualization scripts in sequence."""
    
    print("=" * 70)
    print("PINN Heat Sink Project - Comprehensive Visualization Suite")
    print("=" * 70)
    
    # Ensure plots directory exists
    Path("plots").mkdir(exist_ok=True)
    
    visualization_scripts = [
        # For executives
        ("1. Performance Comparison (Speed Benchmark)", "plot_performance_comparison"),
        
        # For technical audience
        ("2. PINN Architecture Diagram", "plot_pinn_architecture"),
        
        # For domain experts
        ("3. Three-Way Model Comparison", "plot_3way_comparison"),
        
        # For skeptics
        ("4. Enhanced Error Map", "plot_error_map_enhanced"),
        
        # Existing plots
        ("5. Ground Truth Visualization", "plot_ground_truth"),
        ("6. Side-by-Side Prediction", "plot_prediction_vs_truth"),
        ("7. Error Map", "plot_error_map"),
        ("8. Training Loss Curve", "plot_loss_curve"),
        ("9. Workflow Comparison", "plot_workflow_comparison")
    ]
    
    successful_plots = []
    failed_plots = []
    
    for description, module_path in visualization_scripts:
        print(f"\n{description}...")
        try:
            # Import and run the module
            module = __import__(module_path, fromlist=[''])
            
            # Try to call the main function
            # Most plots have 'main' function, some have specific names
            if hasattr(module, 'main'):
                module.main()
            elif hasattr(module, 'create_performance_comparison'):
                module.create_performance_comparison()
            elif hasattr(module, 'create_pinn_architecture_diagram'):
                module.create_pinn_architecture_diagram()
            elif hasattr(module, 'create_3way_comparison'):
                module.create_3way_comparison()
            elif hasattr(module, 'create_enhanced_error_map'):
                module.create_enhanced_error_map()
            elif hasattr(module, 'plot_side_by_side'):
                module.plot_side_by_side()
            elif hasattr(module, 'plot_error_map'):
                module.plot_error_map()
            elif hasattr(module, 'plot_losses'):
                module.plot_losses()
            elif hasattr(module, 'plot_ground_truth_slice'):
                module.plot_ground_truth_slice()
            elif hasattr(module, 'create_workflow_comparison'):
                module.create_workflow_comparison()
            else:
                print(f"⚠️ No main function found in {module_path}")
                continue
            
            successful_plots.append(description)
            print(f"✅ {description} completed successfully")
        except Exception as e:
            print(f"❌ {description} failed: {str(e)}")
            failed_plots.append((description, str(e)))
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Successful: {len(successful_plots)}")
    for plot in successful_plots:
        print(f"   • {plot}")
    
    if failed_plots:
        print(f"\n❌ Failed: {len(failed_plots)}")
        for plot, error in failed_plots:
            print(f"   • {plot}: {error}")
    
    # List generated files
    print(f"\n📁 All plots saved to: plots/")
    plot_files = sorted(Path("plots").glob("*.png"))
    print(f"\nGenerated files ({len(plot_files)} total):")
    for file in plot_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   • {file.name} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review the generated visualizations in the plots/ directory")
    print("2. Use 'performance_comparison.png' for executive presentations")
    print("3. Use 'pinn_architecture.png' to explain the method")
    print("4. Use '3way_comparison.png' to show model improvement")
    print("5. Use 'error_map_enhanced.png' for technical deep-dives")
    print("=" * 70)

if __name__ == "__main__":
    run_all_visualizations()

