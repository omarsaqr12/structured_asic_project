#!/usr/bin/env python3
"""
Generate a histogram showing the distribution of net Half-Perimeter Wire Length (HPWL).

Takes a placement.json file and design netlist to calculate HPWL for each net,
then creates a histogram visualization.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from parse_design import parse_design
from parse_fabric import parse_pins, parse_fabric_cells
from placer import get_port_to_net_mapping


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
                        nets_dict: Dict[int, List[str]],
                        net_to_pins: Optional[Dict[int, List[Tuple[float, float]]]] = None,
                        slot_lookup: Optional[Dict[str, Dict[str, Any]]] = None) -> List[float]:
    """
    Calculate HPWL for all nets in the design.
    
    Args:
        placement: Dict mapping cell_name -> {x, y, fabric_slot_name, ...}
        nets_dict: Dict mapping net_id -> list of cell instances
        net_to_pins: Optional mapping of net_id to fixed pin coordinates
        slot_lookup: Optional dict mapping slot_name -> {x, y, ...} for coordinate lookup
    
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
                
                # If x,y missing but we have slot_lookup and fabric_slot_name, use slot lookup
                if (x is None or y is None) and slot_lookup:
                    slot_name = cell_placement.get('fabric_slot_name')
                    if slot_name and slot_name in slot_lookup:
                        slot = slot_lookup[slot_name]
                        x = slot.get('x')
                        y = slot.get('y')
                
                if x is not None and y is not None:
                    positions.append((x, y))
        
        # Attach any fixed pins on this net
        if net_to_pins and net_id in net_to_pins:
            positions.extend(net_to_pins[net_id])

        # Calculate HPWL for this net
        net_hpwl = calculate_hpwl(positions)
        hpwl_values.append(net_hpwl)
    
    return hpwl_values


def build_net_to_pins_mapping(pins_db: Optional[Dict[str, Any]],
                              port_to_nets: Optional[Dict[str, List[int]]]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Build mapping from net_id to fixed pin coordinates (x_um, y_um).
    """
    if not pins_db or not port_to_nets:
        return {}

    net_to_pins = defaultdict(list)
    for pin in pins_db.get('pins', []):
        if pin.get('status') != 'FIXED':
            continue
        pin_loc = (pin.get('x_um'), pin.get('y_um'))
        if pin_loc[0] is None or pin_loc[1] is None:
            continue

        pin_name = pin.get('name')
        for net_id in port_to_nets.get(pin_name, []):
            net_to_pins[net_id].append(pin_loc)

    return net_to_pins


def plot_net_length_histogram(placement_path: str,
                              design_path: str,
                              output_path: str,
                              bins: int = 50,
                              log_scale: bool = False,
                              placement_name: str = None,
                              pins_path: str = "fabric/pins.yaml",
                              fabric_cells_path: str = "fabric/fabric_cells.yaml"):
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
    
    # Load fabric cells for slot coordinate lookup (optional fallback)
    slot_lookup = None
    if fabric_cells_path and os.path.exists(fabric_cells_path):
        fabric_db = parse_fabric_cells(fabric_cells_path)
        slot_lookup = {}
        for slots in fabric_db.values():
            for slot in slots:
                slot_lookup[slot['name']] = slot
    
    # Load IO pin definitions (optional)
    pins_db = None
    port_to_nets = None
    net_to_pins = {}
    if pins_path and os.path.exists(pins_path):
        print(f"Loading fixed pins from {pins_path}...")
        pins_db = parse_pins(pins_path)
        port_to_nets = get_port_to_net_mapping(design_path)
        net_to_pins = build_net_to_pins_mapping(pins_db, port_to_nets)
        print(f"Mapped {len(net_to_pins)} nets to fixed IO pins")
    else:
        if pins_path:
            print(f"WARNING: Pins file '{pins_path}' not found. Histogram will exclude IO pins.")
        else:
            print("No pins file provided. Histogram will exclude IO pins.")

    # Calculate HPWL for each net
    print("Calculating HPWL for each net...")
    hpwl_values = calculate_net_hpwls(placement, nets_dict, net_to_pins=net_to_pins, slot_lookup=slot_lookup)

    
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
    stats_text += f'Total HPWL (cells + IO): {total_hpwl:.2f} um\n'
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
    parser.add_argument('--pins', type=str, default='fabric/pins.yaml',
                        help='Path to pins YAML file (default: fabric/pins.yaml)')
    
    args = parser.parse_args()
    
    plot_net_length_histogram(
        args.placement,
        args.design,
        args.output,
        bins=args.bins,
        log_scale=args.log_scale,
        placement_name=args.placement_name,
        pins_path=args.pins
    )


if __name__ == '__main__':
    main()

