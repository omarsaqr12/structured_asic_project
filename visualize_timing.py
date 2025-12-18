#!/usr/bin/env python3
"""
Visualize timing analysis results:
1. Parse setup report and plot histogram of endpoint slacks
2. Parse worst path and draw it on the layout
"""

import argparse
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


def parse_setup_report(setup_rpt_path: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """
    Parse setup timing report to extract:
    - All endpoint slacks (for histogram)
    - Worst path details (lowest slack)
    
    Returns:
        (all_slacks, worst_path_dict)
        worst_path_dict contains: {slack, startpoint, endpoint, cells}
    """
    all_slacks = []
    worst_path = None
    worst_slack = float('inf')
    
    current_path = None
    current_path_cells = []
    current_startpoint = None
    current_endpoint = None
    
    with open(setup_rpt_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for startpoint
        if line.startswith('Startpoint:'):
            # Save previous path if exists
            if current_path is not None and current_path['slack'] is not None:
                all_slacks.append(current_path['slack'])
                if current_path['slack'] < worst_slack:
                    worst_slack = current_path['slack']
                    worst_path = current_path.copy()
            
            # Start new path
            match = re.match(r'Startpoint:\s+(.+)', line)
            if match:
                current_startpoint = match.group(1).split()[0]  # Get instance name
                current_path_cells = [current_startpoint]
                current_path = {
                    'startpoint': current_startpoint,
                    'endpoint': None,
                    'slack': None,
                    'cells': [current_startpoint]
                }
        
        # Check for endpoint
        elif line.startswith('Endpoint:'):
            match = re.match(r'Endpoint:\s+(.+)', line)
            if match:
                current_endpoint = match.group(1).split()[0]  # Get instance name
                if current_path:
                    current_path['endpoint'] = current_endpoint
                    if current_endpoint not in current_path['cells']:
                        current_path['cells'].append(current_endpoint)
        
        # Extract cell names from path description lines
        # Format: "T14Y58__R1_DFBBP_0/Q (sky130_fd_sc_hd__dfbbp_1)"
        elif current_path and ('/' in line or line.startswith('T')):
            # Look for cell instance names (format: T##Y##__R#_CELLNAME)
            cell_matches = re.findall(r'(T\d+Y\d+__R\d+_\w+)', line)
            for cell_name in cell_matches:
                if cell_name not in current_path['cells']:
                    current_path['cells'].append(cell_name)
        
        # Check for slack value
        # Format: "0.29   slack (MET)" or "-0.15   slack (VIOLATED)"
        elif 'slack' in line.lower():
            slack_match = re.search(r'([+-]?\d+\.?\d*)\s+slack', line)
            if slack_match:
                slack_value = float(slack_match.group(1))
                if current_path:
                    current_path['slack'] = slack_value
    
        i += 1
    
    # Don't forget the last path
    if current_path is not None and current_path['slack'] is not None:
        all_slacks.append(current_path['slack'])
        if current_path['slack'] < worst_slack:
            worst_slack = current_path['slack']
            worst_path = current_path.copy()
    
    return all_slacks, worst_path


def parse_def_file(def_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Parse DEF file to extract cell locations.
    
    Format: - INST_NAME CELL_TYPE + FIXED ( X Y ) ORIENT ;
    
    Returns:
        Dict mapping instance_name -> (x, y) in microns
    """
    cell_locations = {}
    
    if not os.path.exists(def_path):
        print(f"Warning: DEF file not found: {def_path}")
        return cell_locations
    
    in_components = False
    
    with open(def_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('COMPONENTS'):
                in_components = True
                continue
            elif line.startswith('END COMPONENTS'):
                in_components = False
                continue
            
            if in_components and line.startswith('-'):
                # Parse: - INST_NAME CELL_TYPE + FIXED ( X Y ) ORIENT ;
                # Example: - T14Y58__R1_DFBBP_0 sky130_fd_sc_hd__dfbbp_1 + FIXED ( 14200 5000 ) N ;
                match = re.match(r'-\s+(\S+)\s+\S+\s+\+\s+FIXED\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', line)
                if match:
                    inst_name = match.group(1)
                    x = float(match.group(2))
                    y = float(match.group(3))
                    cell_locations[inst_name] = (x, y)
    
    return cell_locations


def plot_slack_histogram(slacks: List[float], output_path: str):
    """Plot 1D histogram of endpoint slacks."""
    if not slacks:
        print("Warning: No slack values found. Cannot plot histogram.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(slacks, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Slack (ns)', fontsize=12)
    plt.ylabel('Number of Endpoints', fontsize=12)
    plt.title('Distribution of Setup Timing Slack', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_slack = np.mean(slacks)
    min_slack = np.min(slacks)
    max_slack = np.max(slacks)
    median_slack = np.median(slacks)
    
    stats_text = f'Mean: {mean_slack:.3f} ns\n'
    stats_text += f'Min: {min_slack:.3f} ns\n'
    stats_text += f'Max: {max_slack:.3f} ns\n'
    stats_text += f'Median: {median_slack:.3f} ns\n'
    stats_text += f'Total Paths: {len(slacks)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Histogram saved to: {output_path}")
    plt.close()


def plot_worst_path_on_layout(worst_path: Dict[str, Any], 
                               cell_locations: Dict[str, Tuple[float, float]],
                               def_path: str,
                               output_path: str):
    """
    Plot the worst timing path on the layout.
    Draws a bright red line connecting all cells in the path.
    """
    if not worst_path or not worst_path.get('cells'):
        print("Warning: No worst path data available. Cannot plot path on layout.")
        return
    
    # Get die area from DEF file
    die_area = None
    with open(def_path, 'r') as f:
        for line in f:
            if line.startswith('DIEAREA'):
                # Format: DIEAREA ( 0 0 ) ( X Y ) ;
                match = re.search(r'\(\s*[\d.]+\s+[\d.]+\s+\)\s+\(\s*([\d.]+)\s+([\d.]+)\s+\)', line)
                if match:
                    die_area = (float(match.group(1)), float(match.group(2)))
                    break
    
    if not die_area:
        print("Warning: Could not find DIEAREA in DEF file. Using cell bounds.")
        # Fallback: calculate from cell locations
        if cell_locations:
            xs = [loc[0] for loc in cell_locations.values()]
            ys = [loc[1] for loc in cell_locations.values()]
            die_area = (max(xs) + 1000, max(ys) + 1000)
        else:
            print("Error: No cell locations available.")
            return
    
    # Get path cell locations
    path_cells = worst_path['cells']
    path_coords = []
    missing_cells = []
    
    for cell_name in path_cells:
        if cell_name in cell_locations:
            path_coords.append(cell_locations[cell_name])
        else:
            missing_cells.append(cell_name)
    
    if missing_cells:
        print(f"  Warning: {len(missing_cells)} cells not found in DEF file:")
        for cell in missing_cells[:5]:  # Show first 5
            print(f"    - {cell}")
        if len(missing_cells) > 5:
            print(f"    ... and {len(missing_cells) - 5} more")
    
    if not path_coords:
        print("Error: No path cell locations found. Cannot plot path.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot all cells as small dots (light gray)
    all_x = [loc[0] for loc in cell_locations.values()]
    all_y = [loc[1] for loc in cell_locations.values()]
    if all_x and all_y:
        ax.scatter(all_x, all_y, c='lightgray', s=1, alpha=0.3, label='All Cells')
    
    # Plot path cells as larger dots (blue)
    path_x = [coord[0] for coord in path_coords]
    path_y = [coord[1] for coord in path_coords]
    ax.scatter(path_x, path_y, c='blue', s=50, alpha=0.7, zorder=5, label='Path Cells')
    
    # Draw bright red line connecting path cells
    if len(path_coords) > 1:
        ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.8, zorder=4, label='Worst Path')
        # Make it extra bright red
        ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=1.0, zorder=4, color='#FF0000')
    
    # Mark startpoint and endpoint
    if path_coords:
        # Startpoint (first cell)
        ax.scatter(path_x[0], path_y[0], c='green', s=200, marker='s', 
                  edgecolors='black', linewidths=2, zorder=6, label='Startpoint')
        # Endpoint (last cell)
        ax.scatter(path_x[-1], path_y[-1], c='red', s=200, marker='s', 
                  edgecolors='black', linewidths=2, zorder=6, label='Endpoint')
    
    # Set labels and title
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    slack_value = worst_path.get('slack', 'N/A')
    title = f'Worst Setup Timing Path (Slack: {slack_value:.3f} ns)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Set reasonable axis limits
    if all_x and all_y:
        margin = 1000
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Worst path visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize timing analysis results: slack histogram and worst path on layout'
    )
    parser.add_argument('--design', required=True, help='Design name (e.g., arith, 6502)')
    parser.add_argument('--build-dir', default='build', help='Build directory (default: build)')
    parser.add_argument('--setup-rpt', default=None,
                       help='Path to setup report (default: build/{design}/{design}_setup.rpt)')
    parser.add_argument('--def-file', '--def', dest='def_file', default=None,
                       help='Path to DEF file (default: build/{design}/{design}_routed.def)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for plots (default: build/{design}/)')
    
    args = parser.parse_args()
    
    # Build paths
    build_dir = args.build_dir
    design_name = args.design
    
    if args.setup_rpt:
        setup_rpt_path = args.setup_rpt
    else:
        setup_rpt_path = os.path.join(build_dir, design_name, f"{design_name}_setup.rpt")
    
    if args.def_file:
        def_path = args.def_file
    else:
        def_path = os.path.join(build_dir, design_name, f"{design_name}_routed.def")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(build_dir, design_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Timing Visualization for {design_name}")
    print("=" * 60)
    print(f"Setup Report: {setup_rpt_path}")
    print(f"DEF File: {def_path}")
    print(f"Output Directory: {output_dir}")
    print()
    
    # Parse setup report
    print("Step 1: Parsing setup report...")
    if not os.path.exists(setup_rpt_path):
        print(f"  ✗ ERROR: Setup report not found: {setup_rpt_path}")
        return
    
    all_slacks, worst_path = parse_setup_report(setup_rpt_path)
    print(f"  ✓ Found {len(all_slacks)} timing paths")
    
    if worst_path:
        print(f"  ✓ Worst path: {worst_path['startpoint']} -> {worst_path['endpoint']}")
        print(f"    Slack: {worst_path['slack']:.3f} ns")
        print(f"    Path length: {len(worst_path['cells'])} cells")
    else:
        print("  ⚠ WARNING: No worst path found (may be 'No paths found' in report)")
    print()
    
    # Plot histogram
    if all_slacks:
        print("Step 2: Plotting slack histogram...")
        hist_output = os.path.join(output_dir, f"{design_name}_slack_histogram.png")
        plot_slack_histogram(all_slacks, hist_output)
        print()
    
    # Parse DEF file and plot worst path
    if worst_path and worst_path.get('cells'):
        print("Step 3: Parsing DEF file for cell locations...")
        cell_locations = parse_def_file(def_path)
        print(f"  ✓ Found {len(cell_locations)} cell locations in DEF file")
        print()
        
        print("Step 4: Plotting worst path on layout...")
        path_output = os.path.join(output_dir, f"{design_name}_worst_path.png")
        plot_worst_path_on_layout(worst_path, cell_locations, def_path, path_output)
        print()
    
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

