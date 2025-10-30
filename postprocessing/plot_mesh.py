#!/usr/bin/env python3
"""
Simple mesh plotting utility.

Usage:
    python3 plot_mesh.py mesh_file.json
    python3 plot_mesh.py mesh1.json mesh2.json mesh3.json
"""

import sys
from pathlib import Path

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from mesh_visualizer import VEMMeshVisualizer


def main():
    """Simple command line interface for mesh plotting."""
    
    if len(sys.argv) < 2:
        print("Usage: python3 plot_mesh.py mesh_file.json [mesh_file2.json ...]")
        return
    
    mesh_files = sys.argv[1:]
    visualizer = VEMMeshVisualizer()
    
    if len(mesh_files) == 1:
        # Single mesh
        mesh_file = mesh_files[0]
        print(f"🎨 Plotting mesh: {Path(mesh_file).name}")
        
        visualizer.visualize_mesh(
            mesh_file,
            show_nodes=True,
            show_element_ids=False,
            show_node_ids=False,
            save_plot=True,
            show_plot=False
        )
        
        print(f"✅ Mesh plot saved to data/_output/postprocessing/")
        
    else:
        # Multiple meshes comparison
        print(f"🎨 Comparing {len(mesh_files)} meshes...")
        
        visualizer.compare_meshes(mesh_files, save_plot=True, show_plot=False)
        
        # Also create individual plots
        for mesh_file in mesh_files:
            visualizer.visualize_mesh(
                mesh_file,
                show_nodes=True,
                show_element_ids=False,
                show_node_ids=False,
                save_plot=True,
                show_plot=False
            )
        
        print(f"✅ All {len(mesh_files)} mesh plots saved to data/_output/postprocessing/")


if __name__ == "__main__":
    main()
