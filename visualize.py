#!/usr/bin/env python3
"""
Visualization script for Phase 1.
Generates fabric layout plos showing die, core, pins, and fabric slots.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Tuple, Dict, Any, List
from parse_fabric import parse_fabric_cells, parse_pins
from parse_design import parse_design
from placer import extract_nets, calculate_hpwl, get_port_to_net_mapping


# Color mapping for different cell types
CELL_COLORS = {
    'sky130_fd_sc_hd__nand2_2': '#FFD700',      # Gold/Yellow
    'sky130_fd_sc_hd__or2_2': '#FFA500',        # Orange
    'sky130_fd_sc_hd__clkinv_2': '#87CEEB',     # Sky Blue
    'sky130_fd_sc_hd__clkbuf_4': '#4169E1',     # Royal Blue
    'sky130_fd_sc_hd__dfbbp_1': '#FF69B4',      # Hot Pink
    'sky130_fd_sc_hd__conb_1': '#32CD32',       # Lime Green
    'sky130_fd_sc_hd__tapvpwrvgnd_1': '#808080', # Gray
    'sky130_fd_sc_hd__decap_3': '#DDA0DD',      # Plum
    'sky130_fd_sc_hd__decap_4': '#DDA0DD',      # Plum
    'unknown': '#000000'                        # Black
}

# Cell dimensions from fabric.yaml (site dimensions)
SITE_WIDTH = 0.46  # um (from fabric.yaml)
SITE_HEIGHT = 2.72  # um (from fabric.yaml)

# Cell widths in sites (from fabric.yaml cell_definitions)
CELL_WIDTH_SITES = {
    'sky130_fd_sc_hd__nand2_2': 5,
    'sky130_fd_sc_hd__or2_2': 5,
    'sky130_fd_sc_hd__clkinv_2': 4,
    'sky130_fd_sc_hd__clkbuf_4': 6,
    'sky130_fd_sc_hd__dfbbp_1': 26,
    'sky130_fd_sc_hd__conb_1': 3,
    'sky130_fd_sc_hd__tapvpwrvgnd_1': 1,
    'sky130_fd_sc_hd__decap_3': 3,
    'sky130_fd_sc_hd__decap_4': 4,
}

def get_cell_dimensions(cell_type: str) -> Tuple[float, float]:
    """Get cell width and height in microns."""
    width_sites = CELL_WIDTH_SITES.get(cell_type, 5)  # Default to 5 sites
    width_um = width_sites * SITE_WIDTH
    height_um = SITE_HEIGHT
    return width_um, height_um


def plot_fabric_layout(fabric_cells_path: str, pins_path: str, output_path: str):
    """
    Generate the ground-truth plot of the die, core, pins, and fabric slots.
    """
    # Parse fabric and pins
    fabric_db = parse_fabric_cells(fabric_cells_path)
    pins_db = parse_pins(pins_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    
    # Get die and core dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    core_width = pins_db['core']['width_um']
    core_height = pins_db['core']['height_um']
    core_margin = pins_db['die']['core_margin_um']
    
    # Calculate core position (centered in die with margin)
    core_x = core_margin
    core_y = core_margin
    
    # Draw die boundary
    die_rect = patches.Rectangle(
        (0, 0), die_width, die_height,
        linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
    )
    ax.add_patch(die_rect)
    
    # Draw core boundary
    core_rect = patches.Rectangle(
        (core_x, core_y), core_width, core_height,
        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
    )
    ax.add_patch(core_rect)
    
    # Draw all fabric slots as color-coded rectangles
    for cell_type, slots in fabric_db.items():
        color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
        width_um, height_um = get_cell_dimensions(cell_type)
        
        for slot in slots:
            x = slot['x']
            y = slot['y']
            
            # Draw semi-transparent rectangle for each slot
            # Coordinates are already the lower-left corner (DEF PLACED origin)
            slot_rect = patches.Rectangle(
                (x, y),
                width_um, height_um,
                linewidth=0.1, edgecolor='black', facecolor=color, alpha=0.5
            )
            ax.add_patch(slot_rect)
    
    # Draw pins
    for pin in pins_db['pins']:
        x = pin['x_um']
        y = pin['y_um']
        direction = pin['direction']
        
        # Color pins by direction
        pin_color = 'green' if direction == 'INPUT' else 'red'
        
        # Draw pin as a small circle
        pin_circle = plt.Circle((x, y), radius=2, color=pin_color, zorder=10)
        ax.add_patch(pin_circle)
        
        # Optionally label pins (can be commented out if too cluttered)
        # ax.text(x, y, pin['name'], fontsize=4, ha='center', va='center')
    
    # Set axis properties
    ax.set_xlim(-50, die_width + 50)
    ax.set_ylim(-50, die_height + 50)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    ax.set_title('Fabric Layout: Die, Core, Pins, and Fabric Slots', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend for cell types
    legend_elements = []
    for cell_type, color in CELL_COLORS.items():
        if cell_type != 'unknown' and any(cell_type in ct for ct in fabric_db.keys()):
            # Extract short name for legend
            short_name = cell_type.replace('sky130_fd_sc_hd__', '')
            legend_elements.append(patches.Patch(facecolor=color, label=short_name, alpha=0.6))
    
    # Add pin legend
    legend_elements.append(patches.Patch(facecolor='green', label='Input Pin', alpha=1.0))
    legend_elements.append(patches.Patch(facecolor='red', label='Output Pin', alpha=1.0))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Fabric layout saved to {output_path}")


# ============================================================================
# Phase 2 Visualizations (Issues #7-8)
# ============================================================================

def parse_placement_map(map_file_path: str) -> Dict[str, str]:
    """
    Parse placement map file (.map format).
    
    Format: One line per logical instance
    Format: logical_instance_name physical_slot_name
    
    Args:
        map_file_path: Path to .map file
    
    Returns:
        placement_map: Dict mapping logical_name -> physical_slot_name
    """
    placement_map = {}
    
    try:
        with open(map_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    logical_name = parts[0]
                    physical_slot_name = parts[1]
                    placement_map[logical_name] = physical_slot_name
    except FileNotFoundError:
        raise FileNotFoundError(f"Placement map file not found: {map_file_path}")
    except Exception as e:
        raise ValueError(f"Error parsing placement map file: {e}")
    
    return placement_map


def calculate_density_grid(placement_map: Dict[str, str],
                           fabric_db: Dict[str, List[Dict[str, Any]]],
                           pins_db: Dict[str, Any],
                           grid_resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate 2D density grid for placed cells.
    
    Args:
        placement_map: Dict mapping logical_name -> physical_slot_name
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
        pins_db: Dict containing die, core, and pins information
        grid_resolution: Number of grid bins per dimension (default: 50)
    
    Returns:
        density_grid: 2D numpy array of cell counts per bin
        x_bins: X-axis bin edges
        y_bins: Y-axis bin edges
    """
    # Get die dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    
    # Calculate grid bin size
    grid_size_x = die_width / grid_resolution
    grid_size_y = die_height / grid_resolution
    
    # Initialize density grid
    density_grid = np.zeros((grid_resolution, grid_resolution), dtype=int)
    
    # Build slot lookup for fast access
    slot_lookup = {}
    for slots in fabric_db.values():
        for slot in slots:
            slot_lookup[slot['name']] = slot
    
    # Count cells in each grid bin
    for logical_name, slot_name in placement_map.items():
        if slot_name in slot_lookup:
            slot = slot_lookup[slot_name]
            x = slot['x']
            y = slot['y']
            
            # Find grid bin (clamp to valid range)
            bin_x = int(x / grid_size_x)
            bin_y = int(y / grid_size_y)
            
            # Clamp to valid indices
            bin_x = max(0, min(bin_x, grid_resolution - 1))
            bin_y = max(0, min(bin_y, grid_resolution - 1))
            
            # Increment density
            density_grid[bin_y, bin_x] += 1  # Note: y is first index for imshow
    
    # Create bin edges for plotting
    x_bins = np.linspace(0, die_width, grid_resolution + 1)
    y_bins = np.linspace(0, die_height, grid_resolution + 1)
    
    return density_grid, x_bins, y_bins


def plot_placement_density(placement_map_path: str,
                           fabric_cells_path: str,
                           pins_path: str,
                           output_path: str,
                           grid_resolution: int = 50):
    """
    Generate placement density heatmap visualization.
    
    Args:
        placement_map_path: Path to placement .map file
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        output_path: Path to output PNG file
        grid_resolution: Number of grid bins per dimension (default: 50)
    """
    # Parse data
    print(f"Loading placement map from: {placement_map_path}")
    placement_map = parse_placement_map(placement_map_path)
    print(f"Loaded {len(placement_map)} placed cells")
    
    print(f"Loading fabric data from: {fabric_cells_path}")
    fabric_db = parse_fabric_cells(fabric_cells_path)
    
    print(f"Loading pins data from: {pins_path}")
    pins_db = parse_pins(pins_path)
    
    # Calculate density grid
    print(f"Calculating density grid (resolution: {grid_resolution}x{grid_resolution})...")
    density_grid, x_bins, y_bins = calculate_density_grid(
        placement_map, fabric_db, pins_db, grid_resolution
    )
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Get die and core dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    core_width = pins_db['core']['width_um']
    core_height = pins_db['core']['height_um']
    core_margin = pins_db['die']['core_margin_um']
    core_x = core_margin
    core_y = core_margin
    
    # Plot density heatmap using imshow
    # Note: imshow expects (y, x) indexing and origin='lower' to match coordinates
    im = ax.imshow(
        density_grid,
        extent=[0, die_width, 0, die_height],
        origin='lower',
        cmap='viridis',
        interpolation='bilinear',
        aspect='equal'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Cell Density (cells per bin)')
    
    # Overlay die boundary
    die_rect = patches.Rectangle(
        (0, 0), die_width, die_height,
        linewidth=2, edgecolor='black', facecolor='none', linestyle='-'
    )
    ax.add_patch(die_rect)
    
    # Overlay core boundary
    core_rect = patches.Rectangle(
        (core_x, core_y), core_width, core_height,
        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
    )
    ax.add_patch(core_rect)
    
    # Set axis properties
    ax.set_xlim(-50, die_width + 50)
    ax.set_ylim(-50, die_height + 50)
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    ax.set_title('Placement Density Heatmap', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics text
    max_density = np.max(density_grid)
    mean_density = np.mean(density_grid)
    total_cells = len(placement_map)
    stats_text = f'Total Cells: {total_cells}\nMax Density: {max_density} cells/bin\nMean Density: {mean_density:.2f} cells/bin'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Placement density heatmap saved to {output_path}")


def calculate_net_hpwls(placement_map: Dict[str, str],
                        netlist_path: str,
                        fabric_cells_path: str) -> List[float]:
    """
    Calculate HPWL for each net in the design.
    
    Args:
        placement_map: Dict mapping logical_name -> physical_slot_name
        netlist_path: Path to design mapped JSON file
        fabric_cells_path: Path to fabric_cells.yaml
    
    Returns:
        net_hpwls: List of HPWL values (one per net) in microns
    """
    # Load data
    fabric_db = parse_fabric_cells(fabric_cells_path)
    _, netlist_graph = parse_design(netlist_path)
    
    # Extract nets using extract_nets() from placer.py
    nets_dict = extract_nets(netlist_graph)
    
    # Build slot lookup for fast coordinate access
    slot_lookup = {}
    for slots in fabric_db.values():
        for slot in slots:
            slot_lookup[slot['name']] = slot
    
    # Calculate HPWL for each net
    net_hpwls = []
    
    for net_id, cell_list in nets_dict.items():
        positions = []
        
        # Get all connected cell instances and their physical coordinates
        for cell_name in cell_list:
            if cell_name in placement_map:
                slot_name = placement_map[cell_name]
                if slot_name in slot_lookup:
                    slot = slot_lookup[slot_name]
                    positions.append((slot['x'], slot['y']))
        
        # Calculate HPWL using calculate_hpwl() from placer.py
        net_hpwl = calculate_hpwl(positions)
        net_hpwls.append(net_hpwl)
    
    return net_hpwls


def plot_net_length_histogram(placement_map_path: str,
                              netlist_path: str,
                              fabric_cells_path: str,
                              output_path: str):
    """
    Generate net length histogram visualization.
    
    Args:
        placement_map_path: Path to placement .map file
        netlist_path: Path to design mapped JSON file
        fabric_cells_path: Path to fabric_cells.yaml
        output_path: Path to output PNG file
    """
    # Parse placement map
    print(f"Loading placement map from: {placement_map_path}")
    placement_map = parse_placement_map(placement_map_path)
    print(f"Loaded {len(placement_map)} placed cells")
    
    # Calculate net HPWLs
    print(f"Calculating net HPWLs from: {netlist_path}")
    net_hpwls = calculate_net_hpwls(placement_map, netlist_path, fabric_cells_path)
    
    if not net_hpwls:
        raise ValueError("No nets found in design")
    
    # Calculate statistics
    net_hpwls_array = np.array(net_hpwls)
    mean_hpwl = np.mean(net_hpwls_array)
    median_hpwl = np.median(net_hpwls_array)
    max_hpwl = np.max(net_hpwls_array)
    min_hpwl = np.min(net_hpwls_array)
    std_hpwl = np.std(net_hpwls_array)
    
    print(f"Net HPWL Statistics:")
    print(f"  Total nets: {len(net_hpwls)}")
    print(f"  Mean: {mean_hpwl:.2f} um")
    print(f"  Median: {median_hpwl:.2f} um")
    print(f"  Max: {max_hpwl:.2f} um")
    print(f"  Min: {min_hpwl:.2f} um")
    print(f"  Std Dev: {std_hpwl:.2f} um")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Determine number of bins (adaptive based on data range)
    data_range = max_hpwl - min_hpwl
    if data_range > 0:
        # Use 50-100 bins, but adjust based on data range
        num_bins = min(100, max(50, int(data_range / 10)))
    else:
        num_bins = 50
    
    # Create histogram
    n, bins, patches = ax.hist(net_hpwls, bins=num_bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add vertical lines for mean and median
    ax.axvline(mean_hpwl, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_hpwl:.2f} um')
    ax.axvline(median_hpwl, color='green', linestyle='--', linewidth=2, label=f'Median: {median_hpwl:.2f} um')
    
    # Set axis properties
    ax.set_xlabel('Net HPWL (microns)', fontsize=12)
    ax.set_ylabel('Number of Nets', fontsize=12)
    ax.set_title('Net Length Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Position legend and statistics box side by side in top right
    # Legend on the left, statistics box on the right
    ax.legend(loc='upper right', bbox_to_anchor=(0.70, 0.98), fontsize=10)
    
    # Add statistics text box on the right side, next to legend
    stats_text = (f'Total Nets: {len(net_hpwls)}\n'
                  f'Mean: {mean_hpwl:.2f} um\n'
                  f'Median: {median_hpwl:.2f} um\n'
                  f'Max: {max_hpwl:.2f} um\n'
                  f'Min: {min_hpwl:.2f} um\n'
                  f'Std Dev: {std_hpwl:.2f} um')
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Consider log scale on y-axis if distribution is highly skewed
    if np.max(n) > 10 * np.mean(n):
        ax.set_yscale('log')
        ax.set_ylabel('Number of Nets (log scale)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Net length histogram saved to {output_path}")


def main():
    import os
    
    parser = argparse.ArgumentParser(description='Visualize fabric layout and placement results')
    parser.add_argument('command', choices=['init', 'interactive', 'density', 'netlength'],
                        help='Visualization command')
    
    # Common arguments
    parser.add_argument('--fabric-cells', default='fabric/fabric_cells.yaml',
                        help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', default='fabric/pins.yaml',
                        help='Path to pins.yaml')
    parser.add_argument('--output', default=None,
                        help='Output PNG file path')
    
    # Design-specific arguments
    parser.add_argument('--design', type=str, default=None,
                        help='Design name (e.g., 6502). Used to construct default paths.')
    
    # Placement map argument (for density and netlength)
    parser.add_argument('--map', type=str, default=None,
                        help='Path to placement .map file')
    
    # Netlist argument (for netlength)
    parser.add_argument('--netlist', type=str, default=None,
                        help='Path to design mapped JSON file')
    
    # Density-specific arguments
    parser.add_argument('--grid-resolution', type=int, default=50,
                        help='Grid resolution for density heatmap (default: 50)')
    
    args = parser.parse_args()
    
    # Handle default paths based on design name
    if args.design:
        if args.output is None:
            if args.command == 'init':
                args.output = 'fabric_layout.png'
            elif args.command == 'density':
                args.output = f'build/{args.design}/{args.design}_density.png'
            elif args.command == 'netlength':
                args.output = f'build/{args.design}/{args.design}_net_length.png'
        
        if args.map is None and args.command in ['density', 'netlength']:
            args.map = f'build/{args.design}/{args.design}.map'
        
        if args.netlist is None and args.command == 'netlength':
            args.netlist = f'designs/{args.design}_mapped.json'
    
    # Execute command
    if args.command == 'init':
        if args.output is None:
            args.output = 'fabric_layout.png'
        # Check if files exist
        if not os.path.exists(args.fabric_cells):
            parser.error(f"Fabric cells file not found: {args.fabric_cells}")
        if not os.path.exists(args.pins):
            parser.error(f"Pins file not found: {args.pins}")
        plot_fabric_layout(args.fabric_cells, args.pins, args.output)
    
    elif args.command == 'interactive':
        parser.error("'interactive' command not yet implemented")
    
    elif args.command == 'density':
        # Validate required arguments
        if args.map is None:
            parser.error("--map is required for 'density' command (or use --design)")
        if args.output is None:
            parser.error("--output is required for 'density' command (or use --design)")
        
        # Check if files exist before calling function
        if not os.path.exists(args.map):
            parser.error(f"Placement map file not found: {args.map}")
        if not os.path.exists(args.fabric_cells):
            parser.error(f"Fabric cells file not found: {args.fabric_cells}")
        if not os.path.exists(args.pins):
            parser.error(f"Pins file not found: {args.pins}")
        
        # Call plot_placement_density with correct paths
        plot_placement_density(
            args.map,              # placement_map_path
            args.fabric_cells,     # fabric_cells_path
            args.pins,             # pins_path
            args.output,           # output_path
            args.grid_resolution   # grid_resolution
        )
    
    elif args.command == 'netlength':
        # Validate required arguments
        if args.map is None:
            parser.error("--map is required for 'netlength' command (or use --design)")
        if args.netlist is None:
            parser.error("--netlist is required for 'netlength' command (or use --design)")
        if args.output is None:
            parser.error("--output is required for 'netlength' command (or use --design)")
        
        # Check if files exist before calling function
        if not os.path.exists(args.map):
            parser.error(f"Placement map file not found: {args.map}")
        if not os.path.exists(args.netlist):
            parser.error(f"Netlist file not found: {args.netlist}")
        if not os.path.exists(args.fabric_cells):
            parser.error(f"Fabric cells file not found: {args.fabric_cells}")
        
        # Call plot_net_length_histogram with correct paths
        plot_net_length_histogram(
            args.map,              # placement_map_path
            args.netlist,          # netlist_path
            args.fabric_cells,     # fabric_cells_path
            args.output            # output_path
        )


if __name__ == '__main__':
    main()

