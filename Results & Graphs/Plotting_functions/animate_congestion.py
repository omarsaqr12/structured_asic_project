#!/usr/bin/env python3
"""
Net Congestion Heatmap Evolution: Creates an animation showing how routing congestion
evolves during placement optimization. Congestion is calculated by counting how many
nets pass through each grid cell based on their bounding boxes.

This script can:
- Visualize congestion from a static placement file
- Integrate with SA animation to show congestion changes during optimization
- Show congestion heatmaps with color-coding by routing demand
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

try:
    import imageio
except ImportError:
    imageio = None

from validator import validate_design
from parse_fabric import parse_fabric_cells, parse_pins
from parse_design import parse_design
from placer import (
    extract_nets, calculate_hpwl,
    calculate_net_hpwl, calculate_total_hpwl, get_port_to_net_mapping
)

# Color mapping for different cell types (for optional cell overlay)
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

SITE_WIDTH = 0.46  # um
SITE_HEIGHT = 2.72  # um

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


def get_net_positions(net_id: int,
                     placement: Dict[str, Dict[str, Any]],
                     nets_dict: Dict[int, List[str]],
                     fabric_db: Dict[str, List[Dict[str, Any]]],
                     pins_db: Optional[Dict[str, Any]] = None,
                     port_to_nets: Optional[Dict[str, List[int]]] = None) -> List[Tuple[float, float]]:
    """
    Get all positions (cells + pins) for a specific net.
    
    Returns:
        List of (x, y) positions in microns
    """
    positions = []
    
    # Build slot lookup
    slot_lookup = {}
    for slots in fabric_db.values():
        for slot in slots:
            slot_lookup[slot['name']] = slot
    
    # Get cell positions
    cell_list = nets_dict.get(net_id, [])
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
    
    # Get pin positions if available
    if pins_db and port_to_nets:
        pins = pins_db.get('pins', [])
        for pin in pins:
            if pin.get('status') == 'FIXED':
                pin_name = pin.get('name')
                if pin_name in port_to_nets:
                    pin_nets = port_to_nets[pin_name]
                    if net_id in pin_nets:
                        x = pin.get('x_um') or pin.get('x')
                        y = pin.get('y_um') or pin.get('y')
                        if x is not None and y is not None:
                            positions.append((x, y))
    
    return positions


def calculate_congestion_grid(placement: Dict[str, Dict[str, Any]],
                              nets_dict: Dict[int, List[str]],
                              fabric_db: Dict[str, List[Dict[str, Any]]],
                              pins_db: Dict[str, Any],
                              port_to_nets: Optional[Dict[str, List[int]]] = None,
                              grid_resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Calculate congestion grid by counting nets passing through each grid cell.
    
    For each net, we use its bounding box (HPWL) to determine which grid cells it passes through.
    Congestion = number of nets whose bounding box overlaps with a grid cell.
    
    Args:
        placement: Dict mapping cell_name -> {x, y, fabric_slot_name, ...}
        nets_dict: Dict mapping net_id -> list of cell instances
        fabric_db: Dict mapping cell_type -> List of {name, x, y, orient}
        pins_db: Pins database with die/core dimensions
        port_to_nets: Optional port-to-net mapping for I/O nets
        grid_resolution: Number of grid bins per dimension (default: 50)
    
    Returns:
        congestion_grid: 2D numpy array of net counts per bin
        x_bins: X-axis bin edges
        y_bins: Y-axis bin edges
        stats: Dict with min, max, avg, total congestion metrics
    """
    # Get die dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    
    # Calculate grid bin size
    grid_size_x = die_width / grid_resolution
    grid_size_y = die_height / grid_resolution
    
    # Initialize congestion grid
    congestion_grid = np.zeros((grid_resolution, grid_resolution), dtype=int)
    
    # Process each net
    for net_id, cell_list in nets_dict.items():
        # Get all positions for this net (cells + pins)
        positions = get_net_positions(net_id, placement, nets_dict, fabric_db, pins_db, port_to_nets)
        
        if len(positions) < 2:
            continue  # Skip nets with < 2 connections
        
        # Calculate bounding box
        x_coords = [x for x, y in positions]
        y_coords = [y for x, y in positions]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # Find grid cells that this bounding box passes through
        # Convert bounding box to grid indices
        min_bin_x = max(0, int(min_x / grid_size_x))
        max_bin_x = min(grid_resolution - 1, int(max_x / grid_size_x))
        min_bin_y = max(0, int(min_y / grid_size_y))
        max_bin_y = min(grid_resolution - 1, int(max_y / grid_size_y))
        
        # Increment congestion for all grid cells in the bounding box
        for bin_y in range(min_bin_y, max_bin_y + 1):
            for bin_x in range(min_bin_x, max_bin_x + 1):
                congestion_grid[bin_y, bin_x] += 1
    
    # Calculate statistics
    non_zero_cells = congestion_grid[congestion_grid > 0]
    stats = {
        'min': int(np.min(congestion_grid)),
        'max': int(np.max(congestion_grid)),
        'avg': float(np.mean(non_zero_cells)) if len(non_zero_cells) > 0 else 0.0,
        'total_nets': len(nets_dict),
        'congested_cells': int(np.sum(congestion_grid > 0)),
        'total_cells': grid_resolution * grid_resolution
    }
    
    # Create bin edges for plotting
    x_bins = np.linspace(0, die_width, grid_resolution + 1)
    y_bins = np.linspace(0, die_height, grid_resolution + 1)
    
    return congestion_grid, x_bins, y_bins, stats


def create_congestion_frame(placement: Dict[str, Dict[str, Any]],
                           fabric_db: Dict[str, List[Dict[str, Any]]],
                           pins_db: Dict[str, Any],
                           netlist_graph: Dict[str, Dict[str, Any]],
                           nets_dict: Dict[int, List[str]],
                           port_to_nets: Optional[Dict[str, List[int]]],
                           phase: str,
                           current_hpwl: float,
                           grid_resolution: int = 50,
                           show_cells: bool = False,
                           show_pins: bool = True,
                           show_core: bool = True) -> np.ndarray:
    """
    Create a single frame showing congestion heatmap.
    
    Returns:
        Image as numpy array
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    
    # Get die and core dimensions
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    core_width = pins_db['core']['width_um']
    core_height = pins_db['core']['height_um']
    core_margin = pins_db['die']['core_margin_um']
    
    core_x = core_margin
    core_y = core_margin
    
    # Draw die boundary
    die_rect = patches.Rectangle(
        (0, 0), die_width, die_height,
        linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
    )
    ax.add_patch(die_rect)
    
    # Draw core boundary
    if show_core:
        core_rect = patches.Rectangle(
            (core_x, core_y), core_width, core_height,
            linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
        )
        ax.add_patch(core_rect)
    
    # Calculate congestion grid
    congestion_grid, x_bins, y_bins, stats = calculate_congestion_grid(
        placement, nets_dict, fabric_db, pins_db, port_to_nets, grid_resolution
    )
    
    # Plot congestion heatmap
    # Use 'hot' colormap: black (low) -> red -> yellow -> white (high)
    im = ax.imshow(
        congestion_grid,
        extent=[0, die_width, 0, die_height],
        origin='lower',
        cmap='hot',
        interpolation='bilinear',
        aspect='equal',
        alpha=0.8
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Net Congestion (Nets per Grid Cell)', rotation=270, labelpad=20, fontsize=12)
    
    # Draw pins if requested
    if show_pins:
        pins = pins_db.get('pins', [])
        for pin in pins:
            if pin.get('status') == 'FIXED':
                x = pin.get('x_um') or pin.get('x')
                y = pin.get('y_um') or pin.get('y')
                pin_name = pin.get('name', 'unknown')
                if x is not None and y is not None:
                    ax.plot(x, y, 'o', color='cyan', markersize=8, markeredgecolor='black', markeredgewidth=1, zorder=5)
                    ax.text(x, y + 5, pin_name, fontsize=8, ha='center', va='bottom', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7), zorder=6)
    
    # Draw cells if requested (light overlay)
    if show_cells:
        for cell_name, cell_placement in placement.items():
            x = cell_placement.get('x')
            y = cell_placement.get('y')
            
            if x is not None and y is not None:
                # Get cell type from netlist_graph
                cell_type = netlist_graph.get(cell_name, {}).get('type', 'unknown')
                color = CELL_COLORS.get(cell_type, '#000000')
                
                ax.plot(x, y, 'o', color=color, markersize=2, alpha=0.3, zorder=3)
    
    # Set axis properties
    ax.set_xlim(0, die_width)
    ax.set_ylim(0, die_height)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    
    # Add title with metrics
    title = f"Net Congestion Heatmap - {phase}\n"
    title += f"HPWL: {current_hpwl:.1f} um | "
    title += f"Max Congestion: {stats['max']} nets | "
    title += f"Avg Congestion: {stats['avg']:.1f} nets | "
    title += f"Congested Cells: {stats['congested_cells']}/{stats['total_cells']} ({100*stats['congested_cells']/stats['total_cells']:.1f}%)"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid resolution info
    ax.text(0.02, 0.98, f"Grid: {grid_resolution}x{grid_resolution}",
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    if imageio is None:
        raise ImportError("imageio is required. Install with: pip install imageio imageio-ffmpeg")
    try:
        import imageio.v2 as imageio_v2
        img = imageio_v2.imread(buf)
    except ImportError:
        img = imageio.imread(buf)
    buf.close()
    plt.close(fig)
    return img


def animate_congestion_evolution(design_name: str,
                                 initial_placement_path: Optional[str],
                                 design_path: str,
                                 fabric_cells_path: str = 'fabric/fabric_cells.yaml',
                                 pins_path: str = 'fabric/pins.yaml',
                                 output_path: Optional[str] = None,
                                 fps: int = 2,
                                 grid_resolution: int = 50,
                                 show_cells: bool = False,
                                 show_pins: bool = True,
                                 frame_interval: int = 10) -> bool:
    """
    Animate congestion evolution during SA by integrating with SA recording.
    
    This function runs SA with recording and creates a congestion animation.
    """
    from animate_sa_placement import simulated_annealing_with_recording
    
    print(f"Loading design: {design_name}")
    print(f"  Design path: {design_path}")
    print(f"  Fabric path: {fabric_cells_path}")
    print(f"  Pins path: {pins_path}")
    
    # Auto-detect initial placement from best folder if not provided
    if initial_placement_path is None:
        # Try best SA placement WITHOUT ECO first (SA needs original netlist cells only)
        best_placement_no_eco = f"build/{design_name}/Best_sa_alpha0.99_moves1000_Tfinal0.001/{design_name}_placement.json"
        if os.path.exists(best_placement_no_eco):
            initial_placement_path = best_placement_no_eco
            print(f"  Auto-detected best placement (no ECO): {initial_placement_path}")
        else:
            # Try best SA placement WITH ECO (will filter out ECO cells)
            best_placement_with_eco = f"build/{design_name}/Best_sa_alpha0.99_moves1000_Tfinal0.001/{design_name}_placement_with_eco.json"
            if os.path.exists(best_placement_with_eco):
                initial_placement_path = best_placement_with_eco
                print(f"  Auto-detected best placement (with ECO, will filter): {initial_placement_path}")
            else:
                # Fallback to default
                default_placement = f"build/{design_name}/{design_name}_placement.json"
                if os.path.exists(default_placement):
                    initial_placement_path = default_placement
                    print(f"  Using default placement: {initial_placement_path}")
                else:
                    print(f"ERROR: Could not find placement file for {design_name}")
                    print(f"  Tried: {best_placement_no_eco}")
                    print(f"  Tried: {best_placement_with_eco}")
                    print(f"  Tried: {default_placement}")
                    return False
    
    # Load data
    fabric_db = parse_fabric_cells(fabric_cells_path)
    pins_db = parse_pins(pins_path)
    logical_db, netlist_graph = parse_design(design_path)
    
    # Load initial placement
    print(f"Loading initial placement from: {initial_placement_path}")
    with open(initial_placement_path, 'r') as f:
        initial_placement_raw = json.load(f)
    
    # Extract nets (must do before filtering placement)
    nets_dict = extract_nets(netlist_graph)
    port_to_nets = get_port_to_net_mapping(design_path) if os.path.exists(design_path) else None
    
    # Filter placement to only include cells that exist in netlist_graph
    # This handles cases where placement includes ECO cells that aren't in the original netlist
    initial_placement = {}
    eco_cells_count = 0
    for cell_name, cell_data in initial_placement_raw.items():
        if cell_name in netlist_graph:
            initial_placement[cell_name] = cell_data
        else:
            eco_cells_count += 1
    
    if eco_cells_count > 0:
        print(f"  Filtered out {eco_cells_count} ECO cells (not in original netlist)")
        print(f"  Using {len(initial_placement)} cells from original netlist for SA")
    
    print(f"  Found {len(nets_dict)} nets")
    print(f"  Found {len(initial_placement)} placed cells for SA optimization")
    
    # Record frames during SA
    recorded_frames = []
    
    print("\nRunning Simulated Annealing with congestion recording...")
    final_placement = simulated_annealing_with_recording(
        initial_placement, fabric_db, netlist_graph, nets_dict,
        pins_db, port_to_nets,
        T_initial=None,  # Auto-calculate
        alpha=0.99,
        T_final=0.001,
        moves_per_temp=1000,
        W_initial=None,  # Auto-calculate
        beta=0.98,
        P_refine=0.7,
        P_explore=0.3,
        recorded_frames=recorded_frames,
        frame_interval=frame_interval
    )
    
    print(f"\nRecorded {len(recorded_frames)} frames")
    
    # Generate frames
    print(f"\nGenerating congestion heatmap frames...")
    print(f"  Grid resolution: {grid_resolution}x{grid_resolution}")
    frames = []
    
    for idx, (placement, phase, temp, hpwl, best_hpwl, accept_rate, window, iteration) in enumerate(recorded_frames):
        frame = create_congestion_frame(
            placement, fabric_db, pins_db, netlist_graph, nets_dict, port_to_nets,
            phase, hpwl, grid_resolution, show_cells, show_pins
        )
        frames.append(frame)
        
        # Show progress
        if len(recorded_frames) <= 50 or (idx + 1) % 5 == 0 or idx == len(recorded_frames) - 1:
            pct = int((idx + 1) / len(recorded_frames) * 100)
            print(f"  Generated {idx + 1}/{len(recorded_frames)} frames ({pct}%) - {phase}, HPWL={hpwl:.1f} um", end='\r')
    
    print()  # New line after progress
    
    # Save animation
    if output_path is None:
        output_dir = f"build/{design_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{design_name}_congestion_evolution.mp4"
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving congestion evolution video to: {output_path}")
    if imageio is None:
        raise ImportError("imageio is required. Install with: pip install imageio imageio-ffmpeg")
    
    try:
        import imageio.v2 as imageio_v2
        imageio_v2.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
    except ImportError:
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
    
    print(f"\n✓ Congestion evolution video saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Total frames: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(frames)/fps:.1f} seconds")
    print(f"  Grid resolution: {grid_resolution}x{grid_resolution}")
    
    return True


def animate_congestion_from_placement(design_name: str,
                                      placement_path: Optional[str] = None,
                                      fabric_cells_path: str = 'fabric/fabric_cells.yaml',
                                      pins_path: str = 'fabric/pins.yaml',
                                      design_path: Optional[str] = None,
                                      output_path: Optional[str] = None,
                                      grid_resolution: int = 50,
                                      show_cells: bool = False,
                                      show_pins: bool = True) -> bool:
    """
    Create a static congestion heatmap from a placement file.
    
    Auto-detects placement file from best SA results if placement_path is None.
    """
    print(f"Loading design: {design_name}")
    
    # Auto-detect placement file
    if placement_path is None:
        # Try best SA placement first
        best_placement = f"build/{design_name}/Best_sa_alpha0.99_moves1000_Tfinal0.001/{design_name}_placement_with_eco.json"
        if os.path.exists(best_placement):
            placement_path = best_placement
            print(f"  Auto-detected best placement: {placement_path}")
        else:
            # Fallback to default
            default_placement = f"build/{design_name}/{design_name}_placement.json"
            if os.path.exists(default_placement):
                placement_path = default_placement
                print(f"  Using default placement: {placement_path}")
            else:
                print(f"ERROR: Could not find placement file for {design_name}")
                return False
    
    # Auto-detect design file
    if design_path is None:
        design_path = f"designs/{design_name}_mapped.json"
        if not os.path.exists(design_path):
            print(f"ERROR: Could not find design file: {design_path}")
            return False
    
    print(f"  Placement path: {placement_path}")
    print(f"  Design path: {design_path}")
    print(f"  Fabric path: {fabric_cells_path}")
    print(f"  Pins path: {pins_path}")
    
    # Load data
    print("\nLoading data...")
    fabric_db = parse_fabric_cells(fabric_cells_path)
    pins_db = parse_pins(pins_path)
    logical_db, netlist_graph = parse_design(design_path)
    
    # Load placement
    with open(placement_path, 'r') as f:
        placement = json.load(f)
    
    # Extract nets
    nets_dict = extract_nets(netlist_graph)
    port_to_nets = get_port_to_net_mapping(design_path) if os.path.exists(design_path) else None
    
    print(f"  Found {len(nets_dict)} nets")
    print(f"  Found {len(placement)} placed cells")
    
    # Calculate total HPWL
    total_hpwl = calculate_total_hpwl(placement, nets_dict, fabric_db, pins_db, port_to_nets)
    print(f"  Total HPWL: {total_hpwl:.1f} um")
    
    # Create frame
    print("\nGenerating congestion heatmap...")
    frame = create_congestion_frame(
        placement, fabric_db, pins_db, netlist_graph, nets_dict, port_to_nets,
        "Final Placement", total_hpwl, grid_resolution, show_cells, show_pins
    )
    
    # Save image
    if output_path is None:
        output_dir = f"build/{design_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{design_name}_congestion.png"
    
    print(f"Saving image to: {output_path}")
    if imageio is None:
        raise ImportError("imageio is required. Install with: pip install imageio imageio-ffmpeg")
    
    try:
        import imageio.v2 as imageio_v2
        imageio_v2.imwrite(output_path, frame)
    except ImportError:
        imageio.imwrite(output_path, frame)
    
    print(f"Congestion heatmap saved: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Animate net congestion heatmap evolution during placement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Static congestion heatmap from best placement
  python animate_congestion.py --design 6502 --static
  
  # Animate congestion during SA
  python animate_congestion.py --design 6502 --animate --initial-placement build/6502/6502_placement.json
  
  # Custom grid resolution
  python animate_congestion.py --design 6502 --static --grid-resolution 100
        """
    )
    
    parser.add_argument('--design', type=str, required=True,
                       help='Design name (e.g., 6502)')
    parser.add_argument('--static', action='store_true',
                       help='Generate static congestion heatmap from placement file')
    parser.add_argument('--animate', action='store_true',
                       help='Animate congestion evolution during SA')
    parser.add_argument('--placement', type=str, default=None,
                       help='Path to placement JSON file (auto-detected if not provided)')
    parser.add_argument('--initial-placement', type=str, default=None,
                       help='Path to initial placement for SA animation (auto-detected from best folder if not provided)')
    parser.add_argument('--design-path', type=str, default=None,
                       help='Path to design mapped JSON file (auto-detected if not provided)')
    parser.add_argument('--fabric-cells', type=str, default='fabric/fabric_cells.yaml',
                       help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', type=str, default='fabric/pins.yaml',
                       help='Path to pins.yaml')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (auto-generated if not provided)')
    parser.add_argument('--grid-resolution', type=int, default=50,
                       help='Grid resolution for congestion calculation (default: 50)')
    parser.add_argument('--show-cells', action='store_true',
                       help='Show cell overlay on congestion heatmap')
    parser.add_argument('--hide-pins', action='store_true',
                       help='Hide I/O pins on visualization')
    parser.add_argument('--fps', type=int, default=2,
                       help='Frames per second for animation (default: 2)')
    parser.add_argument('--frame-interval', type=int, default=10,
                       help='Record frame every N SA iterations (default: 10)')
    
    args = parser.parse_args()
    
    if not args.static and not args.animate:
        print("ERROR: Must specify either --static or --animate")
        return 1
    
    if args.static and args.animate:
        print("ERROR: Cannot specify both --static and --animate")
        return 1
    
    try:
        if args.static:
            success = animate_congestion_from_placement(
                args.design,
                placement_path=args.placement,
                fabric_cells_path=args.fabric_cells,
                pins_path=args.pins,
                design_path=args.design_path,
                output_path=args.output,
                grid_resolution=args.grid_resolution,
                show_cells=args.show_cells,
                show_pins=not args.hide_pins
            )
        else:  # args.animate
            design_path = args.design_path or f"designs/{args.design}_mapped.json"
            
            success = animate_congestion_evolution(
                args.design,
                args.initial_placement,  # Will auto-detect from best folder if None
                design_path,
                fabric_cells_path=args.fabric_cells,
                pins_path=args.pins,
                output_path=args.output,
                fps=args.fps,
                grid_resolution=args.grid_resolution,
                show_cells=args.show_cells,
                show_pins=not args.hide_pins,
                frame_interval=args.frame_interval
            )
        
        return 0 if success else 1
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
