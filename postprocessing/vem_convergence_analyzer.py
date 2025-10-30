#!/usr/bin/env python3
"""
VEM Convergence Analyzer - Single parameterized script for all mesh types.

This script generates L² and H¹ error convergence charts for different mesh types
comparing RK3 and RK4 time integration methods.

Usage:
    python3 vem_convergence_analyzer.py --mesh-type distorted
    python3 vem_convergence_analyzer.py --mesh-type serendipity
    python3 vem_convergence_analyzer.py --mesh-type voronoi
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple


class VEMConvergenceAnalyzer:
    """Unified VEM convergence analyzer for all mesh types."""
    
    def __init__(self, data_dir: str = "../data/_output/parabolic", output_dir: str = "../data/_output/postprocessing"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'lines.linewidth': 3,
            'lines.markersize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def load_mesh_data(self, mesh_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load RK3 and RK4 data for a specific mesh type."""
        
        # Define mesh configurations
        mesh_configs = {
            'distorted': {
                'rk3_files': [
                    'distorted_mesh_h0p250_results.json',
                    'distorted_mesh_h0p125_results.json', 
                    'distorted_mesh_h0p063_results.json'
                ],
                'rk4_files': [
                    'distorted_mesh_h0p250_rk4_results.json',
                    'distorted_mesh_h0p125_rk4_results.json',
                    'distorted_mesh_h0p063_rk4_results.json'
                ],
                'title': 'Distorted Q4 Elements'
            },
            'serendipity': {
                'rk3_files': [
                    'serendipity_mesh_h0p2500_results.json',
                    'serendipity_mesh_h0p1250_results.json',
                    'serendipity_mesh_h0p0625_results.json'
                ],
                'rk4_files': [
                    'serendipity_mesh_h0p2500_rk4_results.json',
                    'serendipity_mesh_h0p1250_rk4_results.json',
                    'serendipity_mesh_h0p0625_rk4_results.json'
                ],
                'title': 'Serendipity Q8 Elements'
            },
            'voronoi': {
                'rk3_files': [
                    'voronoi_mesh_h0p2500_results.json',
                    'voronoi_mesh_h0p1250_results.json',
                    'voronoi_mesh_h0p0625_results.json'
                ],
                'rk4_files': [],  # No RK4 data for Voronoi yet
                'title': 'Voronoi Polygonal Elements'
            }
        }
        
        if mesh_type not in mesh_configs:
            raise ValueError(f"Unknown mesh type: {mesh_type}")
        
        config = mesh_configs[mesh_type]
        
        # Load RK3 data
        rk3_data = self._load_json_files(config['rk3_files'])
        
        # Load RK4 data (if available)
        rk4_data = pd.DataFrame()
        if config['rk4_files']:
            rk4_data = self._load_json_files(config['rk4_files'])
        
        return rk3_data, rk4_data, config['title']
    
    def _load_json_files(self, filenames: List[str]) -> pd.DataFrame:
        """Load and parse JSON result files."""
        data = []
        
        for filename in filenames:
            filepath = self.data_dir / filename
            if not filepath.exists():
                print(f"⚠️  Warning: {filename} not found, skipping...")
                continue
                
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                
                # Extract mesh size from filename
                if 'h0p250' in filename or 'h0p2500' in filename:
                    mesh_label = '6×6'
                elif 'h0p125' in filename or 'h0p1250' in filename:
                    mesh_label = '12×12'
                elif 'h0p063' in filename or 'h0p0625' in filename:
                    mesh_label = '23×23'
                else:
                    mesh_label = 'Unknown'
                
                data.append({
                    'Mesh': mesh_label,
                    'h': result['mesh_info']['mesh_size_h'],
                    'Nodes': result['mesh_info']['nodes'],
                    'Elements': result['mesh_info']['elements'],
                    'L²-Error': result['results']['l2_error'],
                    'H¹-Error': result['results']['h1_error'],
                    'Status': '✅ PASS' if result['results']['test_passed'] else '❌ FAIL'
                })
                
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('h', ascending=False).reset_index(drop=True)
        
        return df
    
    def create_convergence_charts(self, mesh_type: str) -> None:
        """Create L² and H¹ convergence charts for a mesh type."""
        
        print(f"🔍 Generating convergence charts for {mesh_type} meshes...")
        
        # Load data
        rk3_data, rk4_data, mesh_title = self.load_mesh_data(mesh_type)
        
        if rk3_data.empty:
            print(f"❌ No RK3 data found for {mesh_type}")
            return
        
        # Create L² error chart
        self._create_error_chart(rk3_data, rk4_data, mesh_type, mesh_title, 'L²')
        
        # Create H¹ error chart  
        self._create_error_chart(rk3_data, rk4_data, mesh_type, mesh_title, 'H¹')
        
        print(f"✅ Charts generated for {mesh_type} meshes")
    
    def _create_error_chart(self, rk3_data: pd.DataFrame, rk4_data: pd.DataFrame, 
                           mesh_type: str, mesh_title: str, error_type: str) -> None:
        """Create a single error convergence chart."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot RK3 data
        h_rk3 = rk3_data['h'].values
        error_rk3 = rk3_data[f'{error_type}-Error'].values
        
        ax.loglog(h_rk3, error_rk3, 'o-', label=f'RK3 {error_type} Error', 
                 color='#1f77b4', linewidth=3, markersize=10, alpha=0.8)
        
        # Plot RK4 data (if available)
        if not rk4_data.empty:
            h_rk4 = rk4_data['h'].values
            error_rk4 = rk4_data[f'{error_type}-Error'].values
            
            ax.loglog(h_rk4, error_rk4, 's--', label=f'RK4 {error_type} Error',
                     color='#ff7f0e', linewidth=3, markersize=10, alpha=0.8)
        
        # Add theoretical line
        h_min, h_max = min(h_rk3), max(h_rk3)
        h_theory = np.array([h_min, h_max])
        
        if error_type == 'L²':
            # O(h²) theoretical line
            error_ref = min(error_rk3) * 2  # Reference error
            error_theory = error_ref * (h_theory / h_min)**2
            ax.loglog(h_theory, error_theory, '--', linewidth=2.5, 
                     color='red', alpha=0.7, label='O(h²) theoretical')
        else:  # H¹
            # O(h¹) theoretical line
            error_ref = min(error_rk3) * 2  # Reference error
            error_theory = error_ref * (h_theory / h_min)**1
            ax.loglog(h_theory, error_theory, '--', linewidth=2.5, 
                     color='red', alpha=0.7, label='O(h¹) theoretical')
        
        # Formatting
        ax.set_xlabel('Mesh Size h', fontweight='bold')
        ax.set_ylabel(f'{error_type} Error', fontweight='bold')
        ax.set_title(f'VEM {error_type} Error Convergence: {mesh_title}\n'
                    f'Manufactured Solution: u(t,x,y) = exp(t)·sin(πx)·sin(πy)\n'
                    f'RK3 vs RK4 Time Integration', 
                    fontweight='bold', pad=20)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Save chart
        filename = f"{mesh_type}_{error_type.lower()}_convergence.png"
        filepath = self.output_dir / filename
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  📊 {filename}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='VEM Convergence Analyzer')
    parser.add_argument('--mesh-type', 
                       choices=['distorted', 'serendipity', 'voronoi', 'all'],
                       required=True,
                       help='Mesh type to analyze')
    
    args = parser.parse_args()
    
    analyzer = VEMConvergenceAnalyzer()
    
    if args.mesh_type == 'all':
        # Generate charts for all mesh types
        mesh_types = ['distorted', 'serendipity', 'voronoi']
        print("🚀 Generating convergence charts for all mesh types...")
        print("="*60)
        
        for mesh_type in mesh_types:
            analyzer.create_convergence_charts(mesh_type)
            print()
        
        print("🎉 All convergence charts generated successfully!")
        print(f"\n📁 Charts saved to: {analyzer.output_dir}")
        print("\n📊 Generated 6 charts:")
        for mesh_type in mesh_types:
            print(f"  • {mesh_type}_l2_convergence.png")
            print(f"  • {mesh_type}_h1_convergence.png")
    else:
        # Generate charts for specific mesh type
        analyzer.create_convergence_charts(args.mesh_type)
        print(f"\n📁 Charts saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
