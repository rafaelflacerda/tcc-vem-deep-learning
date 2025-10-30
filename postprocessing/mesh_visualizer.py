#!/usr/bin/env python3
"""
General VEM Mesh Visualizer

This script can visualize different types of VEM meshes from JSON files:
- Voronoi polygonal meshes
- Serendipity Q8 quadrilateral meshes  
- Distorted Q4 quadrilateral meshes

Usage:
    python3 mesh_visualizer.py --mesh-file path/to/mesh.json
    python3 mesh_visualizer.py --mesh-file path/to/mesh.json --show-nodes --show-ids
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Any


class VEMMeshVisualizer:
    """General VEM mesh visualizer for all mesh types."""
    
    def __init__(self, output_dir: str = "../data/_output/postprocessing"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.figsize': (12, 10)
        })
    
    def load_mesh(self, mesh_file: str) -> Tuple[np.ndarray, List[List[int]], Dict[str, Any]]:
        """Load mesh from JSON file and return nodes, elements, and metadata."""
        
        with open(mesh_file, 'r') as f:
            data = json.load(f)
        
        # Extract metadata
        metadata = data.get('metadata', {})
        
        # Extract nodes
        nodes_data = data['nodes']
        nodes = np.zeros((len(nodes_data), 2))
        
        for node in nodes_data:
            nodes[node['id']] = [node['x'], node['y']]
        
        # Extract elements
        elements_data = data['elements']
        elements = []
        
        for element in elements_data:
            vertices = element['vertices']
            elements.append(vertices)
        
        return nodes, elements, metadata
    
    def detect_mesh_type(self, metadata: Dict[str, Any], elements: List[List[int]]) -> str:
        """Detect the mesh type from metadata and element structure."""
        
        # Check metadata first
        if 'meshType' in metadata:
            mesh_type = metadata['meshType']
            if mesh_type == 'voronoi':
                return 'voronoi'
            elif mesh_type == 'serendipity_quad':
                return 'serendipity'
            elif mesh_type == 'distorted_quad_grid':
                return 'distorted'
        
        # Fallback: analyze element structure
        if not elements:
            return 'unknown'
        
        # Check vertices per element
        vertices_per_element = [len(elem) for elem in elements]
        min_vertices = min(vertices_per_element)
        max_vertices = max(vertices_per_element)
        avg_vertices = np.mean(vertices_per_element)
        
        if max_vertices > 6:  # Voronoi can have many vertices
            return 'voronoi'
        elif max_vertices == 8 and min_vertices == 8:  # Q8 serendipity
            return 'serendipity'
        elif max_vertices == 4 and min_vertices == 4:  # Q4 distorted
            return 'distorted'
        else:
            return 'mixed'
    
    def visualize_mesh(self, mesh_file: str, show_nodes: bool = True, 
                      show_element_ids: bool = False, show_node_ids: bool = False,
                      save_plot: bool = True, show_plot: bool = False) -> None:
        """Visualize a mesh from JSON file."""
        
        print(f"🔍 Loading mesh: {mesh_file}")
        
        # Load mesh data
        nodes, elements, metadata = self.load_mesh(mesh_file)
        mesh_type = self.detect_mesh_type(metadata, elements)
        
        print(f"📊 Detected mesh type: {mesh_type}")
        print(f"📐 Nodes: {len(nodes)}, Elements: {len(elements)}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Consistent color scheme for all mesh types
        color_schemes = {
            'voronoi': {'face': 'lightblue', 'edge': 'darkblue', 'node': 'darkred'},
            'serendipity': {'face': 'lightblue', 'edge': 'darkblue', 'node': 'darkred'},
            'distorted': {'face': 'lightblue', 'edge': 'darkblue', 'node': 'darkred'},
            'mixed': {'face': 'lightblue', 'edge': 'darkblue', 'node': 'darkred'},
            'unknown': {'face': 'lightblue', 'edge': 'darkblue', 'node': 'darkred'}
        }
        
        colors = color_schemes.get(mesh_type, color_schemes['unknown'])
        
        # Plot elements as polygons
        patches_list = []
        
        for i, element_vertices in enumerate(elements):
            # Get coordinates of element vertices
            element_coords = nodes[element_vertices]
            
            # Create polygon patch
            polygon = patches.Polygon(element_coords, closed=True, 
                                    facecolor=colors['face'], 
                                    edgecolor=colors['edge'],
                                    linewidth=1.5, alpha=0.7)
            patches_list.append(polygon)
            
            # Add element ID at centroid if requested
            if show_element_ids:
                centroid = np.mean(element_coords, axis=0)
                ax.text(centroid[0], centroid[1], str(i), 
                       ha='center', va='center', fontsize=8, 
                       fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add all patches to the plot
        patch_collection = PatchCollection(patches_list, match_original=True)
        ax.add_collection(patch_collection)
        
        # Plot nodes if requested
        if show_nodes:
            ax.scatter(nodes[:, 0], nodes[:, 1], c=colors['node'], 
                      s=30, alpha=0.8, zorder=5, edgecolors='white', linewidth=0.5)
            
            # Add node IDs if requested
            if show_node_ids:
                for i, (x, y) in enumerate(nodes):
                    ax.text(x, y, str(i), ha='center', va='center', 
                           fontsize=6, color='white', fontweight='bold')
        
        # Set equal aspect ratio and limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        
        # Labels and title
        ax.set_xlabel('x', fontweight='bold')
        ax.set_ylabel('y', fontweight='bold')
        
        # No title for cleaner visualization
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colors['face'], 
                         edgecolor=colors['edge'], alpha=0.7, label='Elements')
        ]
        if show_nodes:
            legend_elements.append(
                plt.scatter([], [], c=colors['node'], s=30, alpha=0.8, 
                           edgecolors='white', linewidth=0.5, label='Nodes')
            )
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.0, 1.0), frameon=True, fancybox=True, shadow=True)
        
        # Save plot if requested
        if save_plot:
            # Generate filename from input mesh file
            mesh_path = Path(mesh_file)
            output_filename = f"{mesh_path.stem}_visualization.png"
            output_path = self.output_dir / output_filename
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Mesh visualization saved: {output_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def compare_meshes(self, mesh_files: List[str], save_plot: bool = True, show_plot: bool = False) -> None:
        """Create a side-by-side comparison of multiple meshes."""
        
        n_meshes = len(mesh_files)
        if n_meshes == 0:
            print("❌ No mesh files provided")
            return
        
        # Calculate subplot layout
        cols = min(3, n_meshes)  # Max 3 columns
        rows = (n_meshes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_meshes == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        # Consistent color scheme for all mesh types
        color_schemes = {
            'voronoi': {'face': 'lightblue', 'edge': 'darkblue', 'node': 'darkred'},
            'serendipity': {'face': 'lightblue', 'edge': 'darkblue', 'node': 'darkred'},
            'distorted': {'face': 'lightblue', 'edge': 'darkblue', 'node': 'darkred'}
        }
        
        for i, mesh_file in enumerate(mesh_files):
            ax = axes[i]
            
            print(f"🔍 Processing mesh {i+1}/{n_meshes}: {mesh_file}")
            
            # Load mesh data
            nodes, elements, metadata = self.load_mesh(mesh_file)
            mesh_type = self.detect_mesh_type(metadata, elements)
            colors = color_schemes.get(mesh_type, color_schemes['distorted'])
            
            # Plot elements
            patches_list = []
            for element_vertices in elements:
                element_coords = nodes[element_vertices]
                polygon = patches.Polygon(element_coords, closed=True,
                                        facecolor=colors['face'],
                                        edgecolor=colors['edge'],
                                        linewidth=1.0, alpha=0.7)
                patches_list.append(polygon)
            
            patch_collection = PatchCollection(patches_list, match_original=True)
            ax.add_collection(patch_collection)
            
            # Plot nodes
            ax.scatter(nodes[:, 0], nodes[:, 1], c=colors['node'], 
                      s=15, alpha=0.8, zorder=5, edgecolors='white', linewidth=0.3)
            
            # Set equal aspect and limits
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            
            # No title for cleaner visualization
            ax.grid(True, alpha=0.3)
            
            # Labels only for bottom and left subplots
            if i >= (rows-1) * cols:  # Bottom row
                ax.set_xlabel('x', fontweight='bold')
            if i % cols == 0:  # Left column
                ax.set_ylabel('y', fontweight='bold')
        
        # Hide unused subplots
        for i in range(n_meshes, len(axes)):
            axes[i].set_visible(False)
        
        # No main title for cleaner visualization
        
        plt.tight_layout()
        
        # Save comparison plot
        if save_plot:
            output_filename = "mesh_comparison.png"
            output_path = self.output_dir / output_filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Mesh comparison saved: {output_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def analyze_mesh_properties(self, mesh_file: str) -> Dict[str, Any]:
        """Analyze and print mesh properties."""
        
        nodes, elements, metadata = self.load_mesh(mesh_file)
        mesh_type = self.detect_mesh_type(metadata, elements)
        
        # Basic statistics
        n_nodes = len(nodes)
        n_elements = len(elements)
        
        # Element size analysis
        vertices_per_element = [len(elem) for elem in elements]
        min_vertices = min(vertices_per_element)
        max_vertices = max(vertices_per_element)
        avg_vertices = np.mean(vertices_per_element)
        
        # Mesh size analysis
        h_values = []
        element_areas = []
        
        for element_vertices in elements:
            element_coords = nodes[element_vertices]
            
            # Calculate element diameter (max distance between vertices)
            distances = []
            for i in range(len(element_coords)):
                for j in range(i+1, len(element_coords)):
                    dist = np.linalg.norm(element_coords[i] - element_coords[j])
                    distances.append(dist)
            
            h_values.append(max(distances))
            
            # Calculate element area using shoelace formula
            x = element_coords[:, 0]
            y = element_coords[:, 1]
            area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
            element_areas.append(area)
        
        h_max = max(h_values)
        h_min = min(h_values)
        h_avg = np.mean(h_values)
        
        total_area = sum(element_areas)
        avg_element_area = np.mean(element_areas)
        
        # Domain bounds
        x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
        y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
        
        properties = {
            'mesh_type': mesh_type,
            'n_nodes': n_nodes,
            'n_elements': n_elements,
            'vertices_per_element': {
                'min': min_vertices,
                'max': max_vertices,
                'avg': avg_vertices
            },
            'mesh_size': {
                'h_max': h_max,
                'h_min': h_min,
                'h_avg': h_avg
            },
            'areas': {
                'total_area': total_area,
                'avg_element_area': avg_element_area
            },
            'domain': {
                'x_range': [x_min, x_max],
                'y_range': [y_min, y_max]
            },
            'metadata': metadata
        }
        
        # Print analysis
        print("\n" + "="*60)
        print(f"📋 MESH ANALYSIS: {Path(mesh_file).name}")
        print("="*60)
        print(f"🔸 Mesh Type: {mesh_type.upper()}")
        print(f"🔸 Nodes: {n_nodes}")
        print(f"🔸 Elements: {n_elements}")
        print(f"🔸 Vertices per element: {min_vertices}-{max_vertices} (avg: {avg_vertices:.1f})")
        print(f"🔸 Mesh size h: {h_min:.6f} - {h_max:.6f} (avg: {h_avg:.6f})")
        print(f"🔸 Total domain area: {total_area:.6f}")
        print(f"🔸 Average element area: {avg_element_area:.6f}")
        print(f"🔸 Domain: x ∈ [{x_min:.3f}, {x_max:.3f}], y ∈ [{y_min:.3f}, {y_max:.3f}]")
        
        if metadata:
            print(f"🔸 Metadata keys: {list(metadata.keys())}")
        
        print("="*60)
        
        return properties


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='VEM Mesh Visualizer')
    parser.add_argument('--mesh-file', type=str, 
                       help='Path to mesh JSON file')
    parser.add_argument('--mesh-files', nargs='+', 
                       help='Multiple mesh files for comparison')
    parser.add_argument('--show-nodes', action='store_true',
                       help='Show mesh nodes')
    parser.add_argument('--show-element-ids', action='store_true',
                       help='Show element IDs')
    parser.add_argument('--show-node-ids', action='store_true',
                       help='Show node IDs')
    parser.add_argument('--show-plot', action='store_true',
                       help='Display plot interactively')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze mesh properties, do not create plots')
    
    args = parser.parse_args()
    
    visualizer = VEMMeshVisualizer()
    
    if args.mesh_files:
        # Multiple mesh comparison
        print("🚀 Creating mesh comparison...")
        if not args.analyze_only:
            visualizer.compare_meshes(args.mesh_files, save_plot=True, show_plot=args.show_plot)
        
        # Analyze each mesh
        for mesh_file in args.mesh_files:
            visualizer.analyze_mesh_properties(mesh_file)
            
    elif args.mesh_file:
        # Single mesh visualization
        if not args.analyze_only:
            visualizer.visualize_mesh(args.mesh_file, 
                                    show_nodes=args.show_nodes,
                                    show_element_ids=args.show_element_ids,
                                    show_node_ids=args.show_node_ids,
                                    save_plot=True, 
                                    show_plot=args.show_plot)
        
        # Analyze mesh properties
        visualizer.analyze_mesh_properties(args.mesh_file)
    else:
        print("❌ Please provide --mesh-file or --mesh-files")
        parser.print_help()


if __name__ == "__main__":
    main()
