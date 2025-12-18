#!/usr/bin/env python3
"""
Net/HPWL Visualization Animation: Creates an animation showing nets as bounding boxes
with color-coding by HPWL length, demonstrating optimization impact during placement.

This script can:
- Visualize nets from a static placement file
- Integrate with SA animation to show HPWL changes during optimization
- Show net bounding boxes, connections, and HPWL metrics
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
    extract_nets, calculate_hpwl, get_port_to_net_mapping,
    calculate_net_hpwl, calculate_total_hpwl
)

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

def get_cell_dimensions(cell_type: str) -> Tuple[float, float]:
    """Get cell width and height in microns."""
    width_sites = CELL_WIDTH_SITES.get(cell_type, 5)
    width_um = width_sites * SITE_WIDTH
    height_um = SITE_HEIGHT
    return width_um, height_um


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
            slot_name = cell_placement.get('fabric_slot_name')
            
            if slot_name and slot_name in slot_lookup:
                slot = slot_lookup[slot_name]
                positions.append((slot['x'], slot['y']))
            elif 'x' in cell_placement and 'y' in cell_placement:
                positions.append((cell_placement['x'], cell_placement['y']))
    
    # Add I/O pin positions
    if pins_db and port_to_nets:
        for pin in pins_db.get('pins', []):
            if pin.get('status') == 'FIXED':
                pin_name = pin['name']
                if pin_name in port_to_nets:
                    pin_nets = port_to_nets[pin_name]
                    if net_id in pin_nets:
                        positions.append((pin['x_um'], pin['y_um']))
    
    return positions


def calculate_net_bounding_box(positions: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate bounding box for a net (HPWL rectangle).
    
    Returns:
        (min_x, min_y, width, height) or None if no positions
    """
    if not positions:
        return None
    
    x_coords = [x for x, y in positions]
    y_coords = [y for x, y in positions]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return (min_x, min_y, width, height)


def create_net_hpwl_frame(placement: Dict[str, Dict[str, Any]],
                          fabric_db: Dict[str, List[Dict[str, Any]]],
                          pins_db: Dict[str, Any],
                          netlist_graph: Dict[str, Dict[str, Any]],
                          nets_dict: Dict[int, List[str]],
                          port_to_nets: Optional[Dict[str, List[int]]],
                          phase: str,
                          current_hpwl: float,
                          show_cells: bool = True,
                          show_nets: bool = True,
                          show_pins: bool = True,
                          max_nets_to_show: Optional[int] = None,
                          highlight_longest_nets: int = 10,
                          specific_net_id: Optional[int] = None) -> np.ndarray:
    """
    Create a single frame showing placement with net HPWL visualization.
    
    Args:
        placement: Current placement dict
        fabric_db: Fabric database
        pins_db: Pins database
        netlist_graph: Netlist graph
        nets_dict: Dict mapping net_id -> list of cells
        port_to_nets: Port to nets mapping
        phase: Description of current phase
        current_hpwl: Current total HPWL
        show_cells: Whether to draw cells
        show_nets: Whether to draw net bounding boxes
        show_pins: Whether to draw I/O pins
        max_nets_to_show: Limit number of nets to show (None = all)
        highlight_longest_nets: Number of longest nets to highlight
    
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
    core_rect = patches.Rectangle(
        (core_x, core_y), core_width, core_height,
        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
    )
    ax.add_patch(core_rect)
    
    # Background for core area
    ax.add_patch(patches.Rectangle(
        (core_x, core_y), core_width, core_height,
        linewidth=0, facecolor='lightgray', alpha=0.1
    ))
    
    # Calculate HPWL for all nets
    net_hpwls = {}
    net_positions_dict = {}
    
    for net_id in nets_dict.keys():
        positions = get_net_positions(net_id, placement, nets_dict, fabric_db, pins_db, port_to_nets)
        if positions:
            net_hpwl = calculate_hpwl(positions)
            net_hpwls[net_id] = net_hpwl
            net_positions_dict[net_id] = positions
    
    # Sort nets by HPWL (longest first)
    sorted_nets = sorted(net_hpwls.items(), key=lambda x: x[1], reverse=True)
    
    # Determine color scale based on HPWL range
    if net_hpwls:
        min_hpwl = min(net_hpwls.values())
        max_hpwl = max(net_hpwls.values())
    else:
        min_hpwl = 0
        max_hpwl = 1
    
    # Create colormap: green (short) -> yellow -> red (long)
    colors = ['#00FF00', '#FFFF00', '#FF0000']  # Green -> Yellow -> Red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('hpwl', colors, N=n_bins)
    
    # Draw net bounding boxes
    if show_nets and net_hpwls:
        # If specific_net_id is provided, show only that net
        if specific_net_id is not None:
            if specific_net_id in net_hpwls:
                nets_to_draw = [(specific_net_id, net_hpwls[specific_net_id])]
            else:
                nets_to_draw = []
        else:
            # Limit number of nets to show if specified
            nets_to_draw = sorted_nets
            if max_nets_to_show:
                nets_to_draw = sorted_nets[:max_nets_to_show]
        
        for net_id, net_hpwl in nets_to_draw:
            positions = net_positions_dict[net_id]
            if len(positions) < 2:
                continue  # Skip nets with < 2 connections
            
            bbox = calculate_net_bounding_box(positions)
            if bbox is None:
                continue
            
            min_x, min_y, width, height = bbox
            
            # Normalize HPWL for color mapping
            if max_hpwl > min_hpwl:
                normalized_hpwl = (net_hpwl - min_hpwl) / (max_hpwl - min_hpwl)
            else:
                normalized_hpwl = 0.5
            
            # Get color from colormap
            color = cmap(normalized_hpwl)
            
            # Highlight longest nets with thicker borders
            is_longest = net_id in [n[0] for n in sorted_nets[:highlight_longest_nets]]
            linewidth = 2.0 if is_longest else 0.5
            alpha = 0.4 if is_longest else 0.2
            
            # Draw bounding box
            rect = patches.Rectangle(
                (min_x, min_y), width, height,
                linewidth=linewidth, edgecolor=color, facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            
            # Draw connections between cells (optional, can be slow for many nets)
            if len(positions) <= 10:  # Only for small nets
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        x1, y1 = positions[i]
                        x2, y2 = positions[j]
                        ax.plot([x1, x2], [y1, y2], 
                               color=color, linewidth=0.3, alpha=0.3, zorder=1)
    
    # Draw placed cells
    if show_cells:
        for cell_name, cell_data in placement.items():
            if cell_name not in netlist_graph:
                continue
            cell_type = netlist_graph[cell_name]['type']
            color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
            width_um, height_um = get_cell_dimensions(cell_type)
            
            x = cell_data.get('x', 0)
            y = cell_data.get('y', 0)
            
            rect = patches.Rectangle(
                (x - width_um/2, y - height_um/2),
                width_um, height_um,
                linewidth=0.5, edgecolor='black', facecolor=color, alpha=0.8, zorder=3
            )
            ax.add_patch(rect)
    
    # Draw pins
    if show_pins:
        for pin in pins_db.get('pins', []):
            if pin.get('status') == 'FIXED':
                ax.plot(pin['x_um'], pin['y_um'], 'ro', markersize=6, alpha=0.8, zorder=4)
    
    # Set axis properties
    ax.set_xlim(0, die_width)
    ax.set_ylim(0, die_height)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    
    # Add title with HPWL metrics
    num_nets = len(net_hpwls)
    avg_hpwl = sum(net_hpwls.values()) / num_nets if num_nets > 0 else 0
    max_net_hpwl = max(net_hpwls.values()) if net_hpwls else 0
    min_net_hpwl = min(net_hpwls.values()) if net_hpwls else 0
    
    if specific_net_id is not None and specific_net_id in net_hpwls:
        # Show info for specific net
        net_hpwl = net_hpwls[specific_net_id]
        # Find rank of this net
        rank = next((i+1 for i, (nid, _) in enumerate(sorted_nets) if nid == specific_net_id), 0)
        title = f'Net/HPWL Visualization - {phase}\n'
        title += f'Showing Net #{rank} (ID: {specific_net_id}) | HPWL: {net_hpwl:.2f} um\n'
        title += f'Total HPWL: {current_hpwl:.2f} um | Total Nets: {num_nets}'
    else:
        title = f'Net/HPWL Visualization - {phase}\n'
        title += f'Total HPWL: {current_hpwl:.2f} um | '
        title += f'Net Count: {num_nets} | '
        title += f'Avg HPWL: {avg_hpwl:.2f} um\n'
        title += f'Min HPWL: {min_net_hpwl:.2f} um | Max HPWL: {max_net_hpwl:.2f} um'
        if max_nets_to_show:
            title += f' | Showing top {max_nets_to_show} nets'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    if net_hpwls and show_nets:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_hpwl, vmax=max_hpwl))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Net HPWL (microns)', rotation=270, labelpad=20, fontsize=10)
    
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


def animate_net_hpwl_from_placement(design_name: str,
                                    placement_path: Optional[str] = None,
                                    fabric_cells_path: str = 'fabric/fabric_cells.yaml',
                                    pins_path: str = 'fabric/pins.yaml',
                                    output_path: Optional[str] = None,
                                    fps: int = 2,
                                    show_cells: bool = True,
                                    show_nets: bool = True,
                                    show_pins: bool = True,
                                    max_nets_to_show: Optional[int] = None,
                                    highlight_longest_nets: int = 10) -> bool:
    """
    Create a static visualization of nets/HPWL from a placement file.
    
    Args:
        design_name: Name of the design (e.g., '6502')
        placement_path: Path to placement JSON file (if None, tries best placement first)
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        output_path: Path to save MP4 video (or PNG if single frame)
        fps: Frames per second (not used for single frame)
        show_cells: Whether to draw cells
        show_nets: Whether to draw net bounding boxes
        show_pins: Whether to draw I/O pins
        max_nets_to_show: Limit number of nets to show (None = all)
        highlight_longest_nets: Number of longest nets to highlight
    
    Returns:
        True if successful, False otherwise
    """
    design_path = f'designs/{design_name}_mapped.json'
    
    if not os.path.exists(design_path):
        print(f"ERROR: Design file not found: {design_path}")
        return False
    
    # Determine placement path - try best folder first
    if placement_path is None:
        # Try to use the "best" placement file first
        best_placement = f'build/{design_name}/Best_sa_alpha0.99_moves1000_Tfinal0.001/{design_name}_placement.json'
        best_placement_eco = f'build/{design_name}/Best_sa_alpha0.99_moves1000_Tfinal0.001/{design_name}_placement_with_eco.json'
        default_placement = f'build/{design_name}/{design_name}_placement.json'
        
        if os.path.exists(best_placement_eco):
            placement_path = best_placement_eco
            print(f"Using best placement file (with ECO): {placement_path}")
        elif os.path.exists(best_placement):
            placement_path = best_placement
            print(f"Using best placement file: {placement_path}")
        elif os.path.exists(default_placement):
            placement_path = default_placement
            print(f"Using default placement file: {placement_path}")
        else:
            print(f"ERROR: No placement file found. Tried:")
            print(f"  - {best_placement_eco}")
            print(f"  - {best_placement}")
            print(f"  - {default_placement}")
            return False
    
    if not os.path.exists(placement_path):
        print(f"ERROR: Placement file not found: {placement_path}")
        return False
    
    # Validate design
    print("Validating design...")
    if not validate_design(fabric_cells_path, design_path):
        print("ERROR: Design validation failed.")
        return False
    
    # Parse data
    print("Loading fabric and design data...")
    fabric_db = parse_fabric_cells(fabric_cells_path)
    logical_db, netlist_graph = parse_design(design_path)
    pins_db = parse_pins(pins_path)
    port_to_nets = get_port_to_net_mapping(design_path)
    nets_dict = extract_nets(netlist_graph)
    
    # Load placement
    print(f"Loading placement from: {placement_path}")
    with open(placement_path, 'r') as f:
        placement = json.load(f)
    
    # Calculate total HPWL
    print("Calculating HPWL...")
    total_hpwl = calculate_total_hpwl(placement, nets_dict, fabric_db, pins_db, port_to_nets)
    print(f"Total HPWL: {total_hpwl:.2f} um")
    
    # Calculate HPWL for all nets and sort
    print("Calculating individual net HPWLs...")
    net_hpwls = {}
    for net_id in nets_dict.keys():
        positions = get_net_positions(net_id, placement, nets_dict, fabric_db, pins_db, port_to_nets)
        if positions and len(positions) >= 2:
            net_hpwl = calculate_hpwl(positions)
            net_hpwls[net_id] = net_hpwl
    
    # Sort nets by HPWL (longest first)
    sorted_nets = sorted(net_hpwls.items(), key=lambda x: x[1], reverse=True)
    print(f"Found {len(sorted_nets)} nets with 2+ connections")
    
    # Determine output path
    if output_path is None:
        output_dir = f'build/{design_name}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{design_name}_net_hpwl_visualization.png'
    
    # Always save individual images for each net (one per net, starting with biggest)
    # Determine output directory
    if output_path and (output_path.endswith('/') or (os.path.isdir(output_path) if os.path.exists(output_path) else False)):
        output_dir = output_path
    elif output_path:
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else f'build/{design_name}'
    else:
        output_dir = f'build/{design_name}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine how many nets to save
    num_nets_to_save = len(sorted_nets)
    if max_nets_to_show:
        num_nets_to_save = min(max_nets_to_show, len(sorted_nets))
    
    print(f"\nGenerating {num_nets_to_save} individual net visualizations...")
    print("  (One image per net, starting with the longest)")
    print("  Optimizing: Pre-computing base elements...")
    
    # Pre-compute all net positions once (already done, but ensure we have them)
    net_positions_dict = {}
    for net_id in nets_dict.keys():
        positions = get_net_positions(net_id, placement, nets_dict, fabric_db, pins_db, port_to_nets)
        if positions and len(positions) >= 2:
            net_positions_dict[net_id] = positions
    
    # Pre-compute cell positions for faster lookup
    cell_positions = {}
    for cell_name, cell_data in placement.items():
        if cell_name in netlist_graph:
            cell_positions[cell_name] = (cell_data.get('x', 0), cell_data.get('y', 0))
    
    # Get die/core dimensions once
    die_width = pins_db['die']['width_um']
    die_height = pins_db['die']['height_um']
    core_width = pins_db['core']['width_um']
    core_height = pins_db['core']['height_um']
    core_margin = pins_db['die']['core_margin_um']
    core_x = core_margin
    core_y = core_margin
    
    # Pre-compute color scale
    if net_hpwls:
        min_hpwl = min(net_hpwls.values())
        max_hpwl = max(net_hpwls.values())
    else:
        min_hpwl = 0
        max_hpwl = 1
    
    colors = ['#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('hpwl', colors, N=100)
    
    saved_count = 0
    
    # Generate images with optimized rendering
    for rank, (net_id, net_hpwl) in enumerate(sorted_nets[:num_nets_to_save], 1):
        # Create optimized frame for this specific net
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        
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
        
        # Background for core area
        ax.add_patch(patches.Rectangle(
            (core_x, core_y), core_width, core_height,
            linewidth=0, facecolor='lightgray', alpha=0.1
        ))
        
        # Draw only cells on this net (much faster than drawing all cells)
        if show_cells and net_id in net_positions_dict:
            positions = net_positions_dict[net_id]
            cells_on_net = set()
            for cell_name in nets_dict.get(net_id, []):
                if cell_name in cell_positions:
                    cells_on_net.add(cell_name)
            
            # Draw cells on this net (highlighted)
            for cell_name in cells_on_net:
                if cell_name in netlist_graph:
                    cell_type = netlist_graph[cell_name]['type']
                    color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
                    width_um, height_um = get_cell_dimensions(cell_type)
                    x, y = cell_positions[cell_name]
                    
                    rect = patches.Rectangle(
                        (x - width_um/2, y - height_um/2),
                        width_um, height_um,
                        linewidth=1.0, edgecolor='black', facecolor=color, alpha=0.9, zorder=3
                    )
                    ax.add_patch(rect)
            
            # Skip drawing other cells for speed (can be enabled if needed for context)
            # Drawing all cells is very slow for large designs
        
        # Draw pins
        if show_pins:
            for pin in pins_db.get('pins', []):
                if pin.get('status') == 'FIXED':
                    ax.plot(pin['x_um'], pin['y_um'], 'ro', markersize=6, alpha=0.8, zorder=4)
        
        # Draw the specific net
        if show_nets and net_id in net_positions_dict:
            positions = net_positions_dict[net_id]
            if len(positions) >= 2:
                bbox = calculate_net_bounding_box(positions)
                if bbox:
                    min_x, min_y, width, height = bbox
                    
                    # Normalize HPWL for color
                    if max_hpwl > min_hpwl:
                        normalized_hpwl = (net_hpwl - min_hpwl) / (max_hpwl - min_hpwl)
                    else:
                        normalized_hpwl = 0.5
                    
                    color = cmap(normalized_hpwl)
                    
                    # Draw bounding box
                    rect = patches.Rectangle(
                        (min_x, min_y), width, height,
                        linewidth=3.0, edgecolor=color, facecolor=color, alpha=0.5, zorder=1
                    )
                    ax.add_patch(rect)
                    
                    # Draw connections
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            x1, y1 = positions[i]
                            x2, y2 = positions[j]
                            ax.plot([x1, x2], [y1, y2], 
                                   color=color, linewidth=1.5, alpha=0.7, zorder=2)
        
        # Set axis properties
        ax.set_xlim(0, die_width)
        ax.set_ylim(0, die_height)
        ax.set_aspect('equal')
        ax.set_xlabel('X (microns)', fontsize=12)
        ax.set_ylabel('Y (microns)', fontsize=12)
        
        # Add title
        title = f'Net/HPWL Visualization - Placement Analysis\n'
        title += f'Showing Net #{rank} (ID: {net_id}) | HPWL: {net_hpwl:.2f} um\n'
        title += f'Total HPWL: {total_hpwl:.2f} um | Total Nets: {len(net_hpwls)}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Save directly to PNG (much faster than imageio conversion)
        output_file = os.path.join(output_dir, f'{design_name}_net_{rank:04d}_id{net_id}_hpwl_{net_hpwl:.1f}um.png')
        fig.savefig(output_file, dpi=75, bbox_inches='tight', format='png', facecolor='white')  # Lower DPI for speed
        plt.close(fig)  # Important: close figure to free memory immediately
        
        saved_count += 1
        if saved_count % 10 == 0 or saved_count == num_nets_to_save:
            print(f"  Saved {saved_count}/{num_nets_to_save} images (Rank {rank}: Net {net_id}, HPWL={net_hpwl:.1f}um)", end='\r')
    
    print()  # New line
    print(f"✓ Saved {saved_count} individual net images to: {output_dir}")
    print(f"  First image: {design_name}_net_0001_... (biggest net)")
    print(f"  Last image: {design_name}_net_{saved_count:04d}_... (smallest net shown)")
    return True
    


def animate_net_hpwl_with_sa(design_name: str,
                             fabric_cells_path: str = 'fabric/fabric_cells.yaml',
                             pins_path: str = 'fabric/pins.yaml',
                             initial_placement_path: Optional[str] = None,
                             output_path: Optional[str] = None,
                             fps: int = 5,
                             T_initial: Optional[float] = None,
                             alpha: float = 0.95,
                             T_final: float = 0.1,
                             moves_per_temp: int = 100,
                             W_initial: Optional[float] = None,
                             beta: float = 0.98,
                             P_refine: float = 0.7,
                             P_explore: float = 0.3,
                             frame_interval: int = 10,
                             show_cells: bool = True,
                             show_nets: bool = True,
                             max_nets_to_show: Optional[int] = 200,
                             highlight_longest_nets: int = 10) -> bool:
    """
    Create an animation of net/HPWL visualization during SA optimization.
    
    This integrates with the SA placement animation to show how nets change.
    """
    from animate_sa_placement import simulated_annealing_with_recording
    from placer import place_design
    
    design_path = f'designs/{design_name}_mapped.json'
    
    if not os.path.exists(design_path):
        print(f"ERROR: Design file not found: {design_path}")
        return False
    
    # Validate design
    print("Validating design...")
    if not validate_design(fabric_cells_path, design_path):
        print("ERROR: Design validation failed.")
        return False
    
    # Parse data
    print("Loading fabric and design data...")
    fabric_db = parse_fabric_cells(fabric_cells_path)
    logical_db, netlist_graph = parse_design(design_path)
    pins_db = parse_pins(pins_path)
    port_to_nets = get_port_to_net_mapping(design_path)
    nets_dict = extract_nets(netlist_graph)
    
    # Get initial placement
    if initial_placement_path and os.path.exists(initial_placement_path):
        print(f"Loading initial placement from: {initial_placement_path}")
        with open(initial_placement_path, 'r') as f:
            initial_placement = json.load(f)
    else:
        print("Running greedy placement to get initial solution...")
        initial_placement = place_design(fabric_cells_path, design_path, pins_path)
        if initial_placement is None:
            print("ERROR: Greedy placement failed.")
            return False
        print("Greedy placement complete.")
    
    # Initialize frame recording
    recorded_frames = []
    
    # Determine output path
    if output_path is None:
        output_dir = f'build/{design_name}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{design_name}_net_hpwl_animation.mp4'
    
    # Ensure output path has .mp4 extension
    if not output_path.lower().endswith('.mp4'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
    
    print(f"\nStarting simulated annealing optimization with net visualization...")
    print(f"Video will be saved to: {output_path}\n")
    
    # Run SA with recording (reuse from animate_sa_placement)
    sa_placement = simulated_annealing_with_recording(
        initial_placement,
        fabric_db,
        netlist_graph,
        nets_dict,
        pins_db,
        port_to_nets,
        T_initial=T_initial,
        alpha=alpha,
        T_final=T_final,
        moves_per_temp=moves_per_temp,
        generate_move_func=None,
        W_initial=W_initial,
        beta=beta,
        P_refine=P_refine,
        P_explore=P_explore,
        recorded_frames=recorded_frames,
        frame_interval=frame_interval
    )
    
    print(f"\nGenerating video from {len(recorded_frames)} frames with net visualization...")
    
    # Check for imageio
    if imageio is None:
        print("ERROR: imageio required for video export.")
        print("Install with: pip install imageio imageio-ffmpeg")
        return False
    
    # Generate frames with net visualization
    print("  Generating frames...")
    frames_images = []
    for i, (placement_state, phase, temp, curr_hpwl, best_hpwl, accept_rate, window, iter_num) in enumerate(recorded_frames):
        frame_img = create_net_hpwl_frame(
            placement_state, fabric_db, pins_db, netlist_graph, nets_dict, port_to_nets,
            phase, curr_hpwl,
            show_cells, show_nets, True, max_nets_to_show, highlight_longest_nets
        )
        frames_images.append(frame_img)
        
        if len(recorded_frames) <= 50 or (i + 1) % 5 == 0 or i == len(recorded_frames) - 1:
            pct = int((i + 1) / len(recorded_frames) * 100)
            print(f"  Generated {i + 1}/{len(recorded_frames)} frames ({pct}%)", end='\r')
    
    print()  # New line
    
    # Save video
    print("Writing video file...")
    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
        for frame in frames_images:
            writer.append_data(frame)
        writer.close()
        
        print(f"\n✓ Video saved successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Error saving video: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create Net/HPWL visualization animation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create static visualization (auto-detects best placement):
  python animate_net_hpwl.py --design 6502
  
  # Create static visualization from specific placement file:
  python animate_net_hpwl.py --design 6502 --placement build/6502/6502_placement.json
  
  # Create animation during SA optimization:
  python animate_net_hpwl.py --design 6502 --sa-animation
  
  # Limit number of nets shown (for performance):
  python animate_net_hpwl.py --design 6502 --max-nets 100
        """
    )
    parser.add_argument('--design', type=str, required=True,
                       help='Design name (e.g., 6502)')
    parser.add_argument('--placement', type=str, default=None,
                       help='Path to placement JSON file (default: tries best placement first, then default)')
    parser.add_argument('--sa-animation', action='store_true',
                       help='Create animation during SA optimization')
    parser.add_argument('--fabric-cells', type=str, default='fabric/fabric_cells.yaml',
                       help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', type=str, default='fabric/pins.yaml',
                       help='Path to pins.yaml')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for video/image')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for video (default: 5)')
    parser.add_argument('--max-nets', type=int, default=None,
                       help='Maximum number of nets to show. If set to 1 or None with multiple nets, saves individual images (default: None)')
    parser.add_argument('--highlight-longest', type=int, default=10,
                       help='Number of longest nets to highlight (default: 10)')
    parser.add_argument('--no-cells', action='store_true',
                       help='Hide cells in visualization')
    parser.add_argument('--no-nets', action='store_true',
                       help='Hide net bounding boxes')
    parser.add_argument('--no-pins', action='store_true',
                       help='Hide I/O pins')
    
    # SA parameters (for --sa-animation)
    parser.add_argument('--T-initial', type=float, default=None,
                       help='Initial temperature (auto-calculated if None)')
    parser.add_argument('--alpha', type=float, default=0.95,
                       help='Cooling rate (default: 0.95)')
    parser.add_argument('--T-final', type=float, default=0.1,
                       help='Final temperature (default: 0.1)')
    parser.add_argument('--moves-per-temp', type=int, default=100,
                       help='Moves per temperature step (default: 100)')
    parser.add_argument('--frame-interval', type=int, default=10,
                       help='Record frame every N iterations (default: 10)')
    
    args = parser.parse_args()
    
    show_cells = not args.no_cells
    show_nets = not args.no_nets
    show_pins = not args.no_pins
    
    if args.sa_animation:
        # Create animation during SA
        success = animate_net_hpwl_with_sa(
            args.design,
            args.fabric_cells,
            args.pins,
            args.placement,
            args.output,
            args.fps,
            args.T_initial,
            args.alpha,
            args.T_final,
            args.moves_per_temp,
            None,  # W_initial
            0.98,  # beta
            0.7,   # P_refine
            0.3,   # P_explore
            args.frame_interval,
            show_cells,
            show_nets,
            args.max_nets,
            args.highlight_longest
        )
    else:
        # Static visualization from placement file (placement_path can be None to auto-detect)
        success = animate_net_hpwl_from_placement(
            args.design,
            args.placement,
            args.fabric_cells,
            args.pins,
            args.output,
            args.fps,
            show_cells,
            show_nets,
            show_pins,
            args.max_nets,
            args.highlight_longest
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
