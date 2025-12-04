#!/usr/bin/env python3
"""
Quick mesh plotting script for the three specific meshes requested.

This script creates visualizations for:
- Voronoi mesh h0p1250
- Serendipity mesh h0p1250  
- Distorted mesh h0p125
"""

import sys
from pathlib import Path

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from mesh_visualizer import VEMMeshVisualizer


def main():
    """Generate visualizations for the three requested meshes."""
    
    print("🎨 VEM Mesh Visualization Suite")
    print("="*50)
    
    # Define the three meshes
    mesh_files = [
        "../data/_voronoi/voronoi_mesh_h0p1250.json",
        "../data/_serendipity/serendipity_mesh_h0p1250.json", 
        "../data/_distorted_mesh/distorted_mesh_h0p125.json"
    ]
    
    mesh_names = [
        "Voronoi Polygonal (h≈0.125)",
        "Serendipity Q8 (h=0.125)",
        "Distorted Q4 (h=0.125)"
    ]
    
    visualizer = VEMMeshVisualizer()
    
    # Create comparison plot
    print("📊 Creating mesh comparison...")
    visualizer.compare_meshes(mesh_files, save_plot=True, show_plot=False)
    
    # Create individual detailed plots
    print("\n📋 Creating individual mesh visualizations...")
    
    for i, (mesh_file, mesh_name) in enumerate(zip(mesh_files, mesh_names)):
        print(f"\n🔍 Processing {mesh_name}...")
        
        # Create detailed visualization without element IDs
        visualizer.visualize_mesh(
            mesh_file,
            show_nodes=True,
            show_element_ids=False,
            show_node_ids=False,
            save_plot=True,
            show_plot=False
        )
        
        # Analyze properties
        properties = visualizer.analyze_mesh_properties(mesh_file)
    
    print("\n🎉 All mesh visualizations completed!")
    print("\n📁 Generated files in data/_output/postprocessing/:")
    print("  📊 mesh_comparison.png (Side-by-side comparison)")
    print("  📊 voronoi_mesh_h0p1250_visualization.png (Detailed Voronoi)")
    print("  📊 serendipity_mesh_h0p1250_visualization.png (Detailed Serendipity)")
    print("  📊 distorted_mesh_h0p125_visualization.png (Detailed Distorted)")
    
    print("\n✨ Key Insights:")
    print("  🌐 Voronoi: True polygonal elements (4-8 vertices)")
    print("  🔳 Serendipity: Regular Q8 elements (8 vertices each)")
    print("  🔄 Distorted: Perturbed Q4 elements (4 vertices each)")


if __name__ == "__main__":
    main()
