#!/usr/bin/env python3
"""
Greedy Placement Animator: Creates an animation of the greedy placement process
to visualize how cells are placed step-by-step.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from validator import validate_design
from parse_fabric import parse_fabric_cells, parse_pins
from parse_design import parse_design
from placer import (
    extract_nets, calculate_hpwl, get_port_to_net_mapping,
    build_net_index, precompute_cell_nets, build_slot_spatial_index,
    find_nearest_slot_linear, find_placed_neighbors_fast, add_io_pin_neighbors,
    calculate_barycenter, get_fallback_position
)

# Color mapping for different cell types (from visualize.py)
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


def place_io_connected_cells_with_snapshots(fabric_db: Dict[str, List[Dict[str, Any]]],
                                            pins_db: Dict[str, Any],
                                            netlist_graph: Dict[str, Dict[str, Any]],
                                            port_to_nets: Dict[str, List[int]],
                                            cell_to_nets: Dict[str, Set[int]],
                                            placement: Dict[str, Dict[str, Any]],
                                            used_slots: Set[str],
                                            slot_lists: Dict,
                                            snapshot_callback) -> int:
    """
    Place I/O-connected cells (seed phase) with snapshot callbacks.
    """
    # Build net-to-pin mapping
    net_to_pin_locations = defaultdict(list)
    for pin in pins_db.get('pins', []):
        if pin.get('status') == 'FIXED':
            pin_name = pin['name']
            if pin_name in port_to_nets:
                for net_id in port_to_nets[pin_name]:
                    net_to_pin_locations[net_id].append((pin['x_um'], pin['y_um']))
    
    # Find all I/O-connected cells
    io_connected_cells = []
    for cell_name in netlist_graph.keys():
        if cell_name in placement:
            continue
        cell_nets = cell_to_nets.get(cell_name, set())
        for net_id in cell_nets:
            if net_id in net_to_pin_locations:
                io_connected_cells.append(cell_name)
                break
    
    cells_placed = 0
    for cell_name in io_connected_cells:
        if cell_name in placement:
            continue
        
        cell_nets = cell_to_nets.get(cell_name, set())
        cell_type = netlist_graph[cell_name]['type']
        
        # Collect pin locations
        pin_locations = []
        for net_id in cell_nets:
            if net_id in net_to_pin_locations:
                pin_locations.extend(net_to_pin_locations[net_id])
        
        if not pin_locations:
            continue
        
        # Calculate barycenter
        target_x = sum(x for x, y in pin_locations) / len(pin_locations)
        target_y = sum(y for x, y in pin_locations) / len(pin_locations)
        
        # Find nearest slot
        available_slots = [slot for slot in fabric_db.get(cell_type, [])
                         if slot['name'] not in used_slots]
        best_slot = find_nearest_slot_linear(target_x, target_y, available_slots)
        
        if best_slot:
            placement[cell_name] = {
                'fabric_slot_name': best_slot['name'],
                'x': best_slot['x'],
                'y': best_slot['y'],
                'orient': best_slot['orient']
            }
            used_slots.add(best_slot['name'])
            cells_placed += 1
            
            # Call snapshot callback
            if snapshot_callback:
                snapshot_callback(placement.copy(), "Seed Phase", cells_placed)
    
    return cells_placed


def place_greedy_barycenter_with_snapshots(fabric_db: Dict[str, List[Dict[str, Any]]],
                                           netlist_graph: Dict[str, Dict[str, Any]],
                                           placement: Dict[str, Dict[str, Any]],
                                           used_slots: Set[str],
                                           net_index: Dict[int, Set[str]],
                                           cell_to_nets: Dict[str, Set[int]],
                                           slot_lists: Dict,
                                           pins_db: Optional[Dict[str, Any]] = None,
                                           port_to_nets: Optional[Dict[str, List[int]]] = None,
                                           snapshot_callback=None,
                                           snapshot_interval: int = 10) -> int:
    """
    Place remaining cells using greedy barycenter with snapshot callbacks.
    """
    unplaced_cells = {cell for cell in netlist_graph.keys() if cell not in placement}
    
    if not unplaced_cells:
        return 0
    
    # Initialize scores
    cell_scores = {}
    fallback_pos = get_fallback_position(pins_db, fabric_db)
    
    for cell in unplaced_cells:
        neighbors = find_placed_neighbors_fast(
            cell, net_index, cell_to_nets[cell], placement
        )
        if pins_db and port_to_nets:
            add_io_pin_neighbors(cell_to_nets[cell], pins_db, port_to_nets, neighbors)
        cell_scores[cell] = (len(neighbors), len(cell_to_nets[cell]))
    
    cells_placed = 0
    while unplaced_cells:
        # Find best cell
        best_cell = max(unplaced_cells, key=lambda c: cell_scores[c])
        cell_type = netlist_graph[best_cell]['type']
        
        # Get neighbors and calculate barycenter
        neighbors = find_placed_neighbors_fast(
            best_cell, net_index, cell_to_nets[best_cell], placement
        )
        
        if pins_db and port_to_nets:
            add_io_pin_neighbors(cell_to_nets[best_cell], pins_db, port_to_nets, neighbors)
        
        if neighbors:
            target_x, target_y = calculate_barycenter(neighbors)
        else:
            target_x, target_y = fallback_pos
        
        # Find nearest slot
        available_slots = [slot for slot in fabric_db.get(cell_type, [])
                         if slot['name'] not in used_slots]
        best_slot = find_nearest_slot_linear(target_x, target_y, available_slots)
        
        if not best_slot:
            unplaced_cells.remove(best_cell)
            del cell_scores[best_cell]
            continue
        
        # Place the cell
        placement[best_cell] = {
            'fabric_slot_name': best_slot['name'],
            'x': best_slot['x'],
            'y': best_slot['y'],
            'orient': best_slot['orient']
        }
        used_slots.add(best_slot['name'])
        unplaced_cells.remove(best_cell)
        del cell_scores[best_cell]
        cells_placed += 1
        
        # Update scores for affected cells
        placed_nets = cell_to_nets[best_cell]
        affected_cells = set()
        for net_id in placed_nets:
            affected_cells.update(net_index.get(net_id, set()))
        
        for affected_cell in affected_cells:
            if affected_cell in unplaced_cells:
                neighbors = find_placed_neighbors_fast(
                    affected_cell, net_index, cell_to_nets[affected_cell], placement
                )
                if pins_db and port_to_nets:
                    add_io_pin_neighbors(cell_to_nets[affected_cell], pins_db, 
                                       port_to_nets, neighbors)
                cell_scores[affected_cell] = (len(neighbors), len(cell_to_nets[affected_cell]))
        
        # Call snapshot callback at intervals
        if snapshot_callback and cells_placed % snapshot_interval == 0:
            snapshot_callback(placement.copy(), "Grow Phase", cells_placed)
    
    # Final snapshot
    if snapshot_callback:
        snapshot_callback(placement.copy(), "Complete", cells_placed)
    
    return cells_placed


def create_placement_frame(placement: Dict[str, Dict[str, Any]],
                           fabric_db: Dict[str, List[Dict[str, Any]]],
                           pins_db: Dict[str, Any],
                           netlist_graph: Dict[str, Dict[str, Any]],
                           phase: str,
                           cells_placed: int,
                           total_cells: int,
                           frame_number: int) -> plt.Figure:
    """
    Create a single frame showing current placement state.
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
    
    # Draw all fabric slots (unplaced) as light gray
    for cell_type, slots in fabric_db.items():
        color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
        width_um, height_um = get_cell_dimensions(cell_type)
        
        for slot in slots:
            x = slot['x']
            y = slot['y']
            rect = patches.Rectangle(
                (x - width_um/2, y - height_um/2),
                width_um, height_um,
                linewidth=0.1, edgecolor='lightgray', facecolor='lightgray', alpha=0.1
            )
            ax.add_patch(rect)
    
    # Draw placed cells
    for cell_name, cell_data in placement.items():
        if cell_name not in netlist_graph:
            continue
        cell_type = netlist_graph[cell_name]['type']
        color = CELL_COLORS.get(cell_type, CELL_COLORS['unknown'])
        width_um, height_um = get_cell_dimensions(cell_type)
        
        x = cell_data['x']
        y = cell_data['y']
        
        rect = patches.Rectangle(
            (x - width_um/2, y - height_um/2),
            width_um, height_um,
            linewidth=0.5, edgecolor='black', facecolor=color, alpha=0.7
        )
        ax.add_patch(rect)
    
    # Draw pins
    for pin in pins_db.get('pins', []):
        if pin.get('status') == 'FIXED':
            ax.plot(pin['x_um'], pin['y_um'], 'ro', markersize=4, alpha=0.6)
    
    # Set axis properties
    ax.set_xlim(0, die_width)
    ax.set_ylim(0, die_height)
    ax.set_aspect('equal')
    ax.set_xlabel('X (microns)', fontsize=12)
    ax.set_ylabel('Y (microns)', fontsize=12)
    
    # Add title with phase and progress
    progress_pct = (cells_placed / total_cells * 100) if total_cells > 0 else 0
    title = f'Greedy Placement Animation - {phase}\n'
    title += f'Frame {frame_number}: {cells_placed}/{total_cells} cells placed ({progress_pct:.1f}%)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig


def animate_greedy_placement(design_name: str,
                            fabric_cells_path: str = 'fabric/fabric_cells.yaml',
                            pins_path: str = 'fabric/pins.yaml',
                            output_path: str = None,
                            snapshot_interval: int = 10,
                            fps: int = 5):
    """
    Create an animation of the greedy placement process.
    
    Args:
        design_name: Name of the design (e.g., '6502')
        fabric_cells_path: Path to fabric_cells.yaml
        pins_path: Path to pins.yaml
        output_path: Output path for GIF (default: build/{design}/greedy_placement_animation.gif)
        snapshot_interval: Number of cells to place between snapshots (default: 10)
        fps: Frames per second for animation (default: 5)
    """
    design_path = f'designs/{design_name}_mapped.json'
    
    if not os.path.exists(design_path):
        print(f"ERROR: Design file not found: {design_path}")
        sys.exit(1)
    
    # Validate design
    print("Validating design...")
    if not validate_design(fabric_cells_path, design_path):
        print("ERROR: Design validation failed.")
        sys.exit(1)
    
    # Parse data
    print("Loading fabric and design data...")
    fabric_db = parse_fabric_cells(fabric_cells_path)
    logical_db, netlist_graph = parse_design(design_path)
    pins_db = parse_pins(pins_path)
    port_to_nets = get_port_to_net_mapping(design_path)
    
    # Build indices
    print("Building optimization indices...")
    net_index = build_net_index(netlist_graph)
    cell_to_nets = precompute_cell_nets(netlist_graph)
    slot_lists = build_slot_spatial_index(fabric_db)
    
    total_cells = len(netlist_graph)
    
    # Initialize placement
    placement = {}
    used_slots = set()
    snapshots = []
    
    def snapshot_callback(placement_state, phase, cells_placed):
        """Callback to capture placement snapshots."""
        snapshots.append((placement_state.copy(), phase, cells_placed))
        print(f"  Snapshot: {phase} - {cells_placed}/{total_cells} cells placed")
    
    # Run placement with snapshots
    print("\nPlacing cells with snapshots...")
    print("Seed Phase: Placing I/O-connected cells...")
    io_cells = place_io_connected_cells_with_snapshots(
        fabric_db, pins_db, netlist_graph, port_to_nets,
        cell_to_nets, placement, used_slots, slot_lists,
        snapshot_callback
    )
    print(f"Placed {io_cells} I/O-connected cells.")
    
    print("Grow Phase: Placing remaining cells...")
    remaining_cells = place_greedy_barycenter_with_snapshots(
        fabric_db, netlist_graph, placement, used_slots,
        net_index, cell_to_nets, slot_lists, pins_db, port_to_nets,
        snapshot_callback, snapshot_interval
    )
    print(f"Placed {remaining_cells} remaining cells.")
    
    print(f"\nTotal snapshots captured: {len(snapshots)}")
    
    # Create output directory
    if output_path is None:
        output_dir = f'build/{design_name}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/greedy_placement_animation.gif'
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate frames
    print(f"\nGenerating {len(snapshots)} frames...")
    frames = []
    for i, (placement_state, phase, cells_placed) in enumerate(snapshots):
        fig = create_placement_frame(
            placement_state, fabric_db, pins_db, netlist_graph,
            phase, cells_placed, total_cells, i
        )
        frames.append(fig)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(snapshots)} frames...")
    
    print("Creating animation...")
    # Save frames as images first, then create GIF
    temp_dir = f'{output_dir}/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_files = []
    for i, fig in enumerate(frames):
        temp_path = f'{temp_dir}/frame_{i:04d}.png'
        fig.savefig(temp_path, dpi=100, bbox_inches='tight')
        temp_files.append(temp_path)
        plt.close(fig)
    
    # Create GIF using imageio or PIL
    try:
        import imageio
        print(f"Writing GIF to {output_path}...")
        images = [imageio.imread(f) for f in temp_files]
        imageio.mimsave(output_path, images, fps=fps, loop=0)
        print(f"Animation saved to {output_path}")
    except ImportError:
        try:
            from PIL import Image
            images = [Image.open(f) for f in temp_files]
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=1000//fps,
                loop=0
            )
            print(f"Animation saved to {output_path}")
        except ImportError:
            print("ERROR: Need imageio or PIL to create GIF. Install with: pip install imageio")
            print(f"Frames saved to {temp_dir}/")
            sys.exit(1)
    
    # Clean up temp files
    import shutil
    shutil.rmtree(temp_dir)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Create animation of greedy placement process'
    )
    parser.add_argument('--design', type=str, required=True,
                       help='Design name (e.g., 6502)')
    parser.add_argument('--fabric-cells', type=str, default='fabric/fabric_cells.yaml',
                       help='Path to fabric_cells.yaml')
    parser.add_argument('--pins', type=str, default='fabric/pins.yaml',
                       help='Path to pins.yaml')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for GIF (default: build/{design}/greedy_placement_animation.gif)')
    parser.add_argument('--snapshot-interval', type=int, default=10,
                       help='Number of cells between snapshots (default: 10)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for animation (default: 5)')
    
    args = parser.parse_args()
    
    animate_greedy_placement(
        args.design,
        args.fabric_cells,
        args.pins,
        args.output,
        args.snapshot_interval,
        args.fps
    )


if __name__ == '__main__':
    main()




