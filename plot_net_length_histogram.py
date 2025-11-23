#!/usr/bin/env python3
"""
Generate a histogram showing the distribution of net Half-Perimeter Wire Length (HPWL).

Takes a placement.json file and design netlist to calculate HPWL for each net,
then creates a histogram visualization.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from parse_design import parse_design


def extract_nets(netlist_graph: Dict[str, Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Extract all unique nets from netlist_graph.
    
    A net is identified by a net ID (integer) that connects multiple cells.
    In Yosys JSON format, connections are stored as lists of net IDs.
    
    Args:
        netlist_graph: Dict mapping instance_name -> {type, connections}
    
    Returns:
        nets_dict: Dict mapping net_id -> list of cell instances connected to this net
    """
    nets_dict = defaultdict(list)
    
    for cell_name, cell_data in netlist_graph.items():
        connections = cell_data.get('connections', {})
        for port_name, net_ids in connections.items():
            # net_ids is a list of integers (net IDs)
            for net_id in net_ids:
                if cell_name not in nets_dict[net_id]:
                    nets_dict[net_id].append(cell_name)
    
    return dict(nets_dict)


def calculate_hpwl(positions: List[Tuple[float, float]]) -> float:
    """
    Calculate Half-Perimeter Wirelength (HPWL) for a single net.
    
    Formula: HPWL = (max_x - min_x) + (max_y - min_y)
    
    Args:
        positions: List of (x, y) coordinates in microns
    
    Returns:
        HPWL in microns (float)
    """
    if len(positions) == 0 or len(positions) == 1:
        return 0.0
    
    if len(positions) == 2:
        x1, y1 = positions[0]
        x2, y2 = positions[1]
        return abs(x2 - x1) + abs(y2 - y1)
    
    # For 3+ cells, find bounding box
    x_coords = [x for x, y in positions]
    y_coords = [y for x, y in positions]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    return (max_x - min_x) + (max_y - min_y)


def calculate_net_hpwls(placement: Dict[str, Dict[str, Any]],
                        nets_dict: Dict[int, List[str]]) -> List[float]:
    """
    Calculate HPWL for all nets in the design.
    
    Args:
        placement: Dict mapping cell_name -> {x, y, ...}
        nets_dict: Dict mapping net_id -> list of cell instances
    
    Returns:
        List of HPWL values (one per net) in microns
    """
    hpwl_values = []
    
    for net_id, cell_list in nets_dict.items():
        positions = []
        
        # Get positions of all cells on this net
        for cell_name in cell_list:
            if cell_name in placement:
                cell_placement = placement[cell_name]
                x = cell_placement.get('x')
                y = cell_placement.get('y')
                
                if x is not None and y is not None:
                    positions.append((x, y))
        
        # Calculate HPWL for this net
        net_hpwl = calculate_hpwl(positions)
        hpwl_values.append(net_hpwl)
    
    return hpwl_values


def plot_net_length_histogram(placement_path: str,
                              design_path: str,
                              output_path: str,
                              bins: int = 50,
                              log_scale: bool = False,
                              placement_name: str = None):
    """
    Generate a histogram of net HPWL values.
    
    Args:
        placement_path: Path to placement.json
        design_path: Path to design netlist JSON
        output_path: Output PNG file path
        bins: Number of histogram bins
        log_scale: Whether to use log scale for x-axis
        placement_name: Name to display for the placement (defaults to filename if not provided)
    """
    # Load placement data
    print(f"Loading placement from {placement_path}...")
    with open(placement_path, 'r') as f:
        placement = json.load(f)
    
    # Load and parse design netlist
    print(f"Loading design netlist from {design_path}...")
    logical_db, netlist_graph = parse_design(design_path)
    
    # Extract nets
    print("Extracting nets from netlist...")
    nets_dict = extract_nets(netlist_graph)
    print(f"Found {len(nets_dict)} nets")
    
    # Calculate HPWL for each net
    print("Calculating HPWL for each net...")
    hpwl_values = calculate_net_hpwls(placement, nets_dict)
    
    # Filter out zero-length nets (single-cell nets or unplaced cells)
    hpwl_values = [hpwl for hpwl in hpwl_values if hpwl > 0]
    
    if not hpwl_values:
        print("WARNING: No nets with non-zero HPWL found!")
        return
    
    # Calculate statistics
    mean_hpwl = np.mean(hpwl_values)
    median_hpwl = np.median(hpwl_values)
    max_hpwl = np.max(hpwl_values)
    min_hpwl = np.min(hpwl_values)
    total_hpwl = np.sum(hpwl_values)
    
    # Use provided placement name or default to filename
    if placement_name is None:
        placement_name = os.path.basename(placement_path)
    
    print(f"HPWL Statistics:")
    print(f"  Total nets: {len(hpwl_values)}")
    print(f"  Total HPWL: {total_hpwl:.2f} um")
    print(f"  Mean HPWL: {mean_hpwl:.2f} um")
    print(f"  Median HPWL: {median_hpwl:.2f} um")
    print(f"  Min HPWL: {min_hpwl:.2f} um")
    print(f"  Max HPWL: {max_hpwl:.2f} um")
    
    # Create histogram
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot histogram
    n, bins, patches = ax.hist(hpwl_values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add vertical lines for mean and median
    ax.axvline(mean_hpwl, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_hpwl:.2f} um')
    ax.axvline(median_hpwl, color='green', linestyle='--', linewidth=2, label=f'Median: {median_hpwl:.2f} um')
    
    # Set labels and title
    ax.set_xlabel('Net HPWL (microns)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Nets', fontsize=12, fontweight='bold')
    ax.set_title(f'Net Length Histogram\nPlacement: {placement_name}\n({len(hpwl_values)} nets, Mean: {mean_hpwl:.2f} um, Median: {median_hpwl:.2f} um)',
                fontsize=14, fontweight='bold')
    
    # Set log scale if requested
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlabel('Net HPWL (microns, log scale)', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Placement: {placement_name}\n'
    stats_text += f'Total Nets: {len(hpwl_values)}\n'
    stats_text += f'Total HPWL: {total_hpwl:.2f} um\n'
    stats_text += f'Mean: {mean_hpwl:.2f} um\n'
    stats_text += f'Median: {median_hpwl:.2f} um\n'
    stats_text += f'Min: {min_hpwl:.2f} um\n'
    stats_text += f'Max: {max_hpwl:.2f} um'
    
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate a histogram of net Half-Perimeter Wire Length (HPWL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python plot_net_length_histogram.py placement.json designs/6502_mapped.json output.png
  
  # With custom bins and log scale
  python plot_net_length_histogram.py placement.json designs/6502_mapped.json output.png --bins 100 --log-scale
  
  # With custom placement name
  python plot_net_length_histogram.py placement.json designs/6502_mapped.json output.png --placement-name "SA alpha=0.99, moves=1000"
        """
    )
    
    parser.add_argument('placement', help='Path to placement.json file')
    parser.add_argument('design', help='Path to design netlist JSON file (e.g., designs/6502_mapped.json)')
    parser.add_argument('output', help='Output PNG file path')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins (default: 50)')
    parser.add_argument('--log-scale', action='store_true', default=False,
                        help='Use logarithmic scale for x-axis')
    parser.add_argument('--placement-name', type=str, default=None,
                        help='Name to display for the placement (defaults to filename if not provided)')
    
    args = parser.parse_args()
    
    plot_net_length_histogram(
        args.placement,
        args.design,
        args.output,
        bins=args.bins,
        log_scale=args.log_scale,
        placement_name=args.placement_name
    )


if __name__ == '__main__':
    main()

